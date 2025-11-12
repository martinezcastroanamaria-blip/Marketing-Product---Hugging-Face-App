import os, re, time, json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from dateutil import parser as dtparser
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from nltk.corpus import stopwords

import gspread
from oauth2client.service_account import ServiceAccountCredentials

# =========================================================
# CONFIG STREAMLIT
# =========================================================
st.set_page_config(page_title="YouTube Ad Analyzer Pro", layout="wide")
st.title("📊 YouTube Ad Analyzer – Marketing Insights Pro")
st.caption("Analyze video performance, audience sentiment and channel history.")

# =========================================================
# NLTK stuff
# =========================================================
try:
   nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
   nltk.download("vader_lexicon")

try:
   nltk.data.find("corpora/stopwords.zip")
except LookupError:
   nltk.download("stopwords")

sia = SentimentIntensityAnalyzer()
STOPWORDS = set(stopwords.words("english"))

# =========================================================
# GOOGLE SHEETS (via HF secret)
# =========================================================
SHEET_NAME = "Marketing dashboard data"

def get_sheet():
   """Obtiene la hoja de Google usando el JSON guardado en el secret HF."""
   scope = [
       "https://spreadsheets.google.com/feeds",
       "https://www.googleapis.com/auth/spreadsheets",
       "https://www.googleapis.com/auth/drive.file",
       "https://www.googleapis.com/auth/drive",
   ]

   raw_json = os.getenv("GOOGLE_CREDENTIALS_JSON")
   if not raw_json:
       st.error("GOOGLE_CREDENTIALS_JSON secret not found. Add it in HF -> Settings -> Variables.")
       raise RuntimeError("Missing GOOGLE_CREDENTIALS_JSON")

   info = json.loads(raw_json)
   creds = ServiceAccountCredentials.from_json_keyfile_dict(info, scope)
   client = gspread.authorize(creds)
   sheet = client.open(SHEET_NAME).sheet1
   return sheet

EXPECTED_COLS = ["video_id", "title", "channel", "engagement_rate", "date"]

def load_portfolio_from_sheet():
   try:
       sheet = get_sheet()
       values = sheet.get_all_values()

       if not values:
           return []

       header = values[0]

       if set(EXPECTED_COLS).issubset(set(header)):
           rows = sheet.get_all_records()
           return rows

       fixed = []
       for row in values:
           fixed_row = row + [""] * (len(EXPECTED_COLS) - len(row))
           fixed.append(dict(zip(EXPECTED_COLS, fixed_row[:len(EXPECTED_COLS)])))
       return fixed

   except Exception as e:
       st.warning(f"⚠️ Could not load data from Google Sheets: {e}")
       return []

def append_to_sheet(row_dict: dict):
   try:
       sheet = get_sheet()
       cols = ["video_id", "title", "channel", "engagement_rate", "date"]
       row = [row_dict.get(c, "") for c in cols]
       sheet.append_row(row)
   except Exception as e:
       st.error(f"❌ Could not save to Google Sheets: {e}")

# =========================================================
# SESSION STATE
# =========================================================
if "portfolio" not in st.session_state:
   st.session_state["portfolio"] = load_portfolio_from_sheet()

if "last_analysis" not in st.session_state:
   st.session_state["last_analysis"] = None

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def extract_video_id(url_or_id):
   if re.match(r"^[\w-]{11}$", url_or_id):
       return url_or_id
   m = re.search(r"v=([\w-]{11})", url_or_id) or re.search(r"youtu\.be/([\w-]{11})", url_or_id)
   return m.group(1) if m else ""

def iso8601_to_hms(iso):
   h = re.search(r"(\d+)H", iso or "")
   m = re.search(r"(\d+)M", iso or "")
   s = re.search(r"(\d+)S", iso or "")
   hh, mm, ss = int(h.group(1)) if h else 0, int(m.group(1)) if m else 0, int(s.group(1)) if s else 0
   return f"{hh:02d}:{mm:02d}:{ss:02d}"

def clean_html(t):
   return BeautifulSoup(t or "", "html.parser").get_text(" ", strip=True)

def days_since(iso_date):
   try:
       dt = dtparser.isoparse(iso_date).astimezone(timezone.utc)
       return max((datetime.now(timezone.utc) - dt).days, 1)
   except Exception:
       return np.nan

# =========================================================
# YOUTUBE API
# =========================================================
def get_video_details(api_key, vid):
   yt = build("youtube", "v3", developerKey=api_key)
   r = yt.videos().list(part="snippet,statistics,contentDetails", id=vid).execute()
   if not r.get("items"):
       return None
   it = r["items"][0]
   sn, stt, cd = it.get("snippet", {}), it.get("statistics", {}), it.get("contentDetails", {})
   return {
       "video_id": it["id"],
       "title": sn.get("title", ""),
       "channel_title": sn.get("channelTitle", ""),
       "published_at": sn.get("publishedAt", ""),
       "tags": sn.get("tags", []),
       "thumbnail": sn.get("thumbnails", {}).get("high", {}).get("url"),
       "views": int(stt.get("viewCount", 0)),
       "likes": int(stt.get("likeCount", 0)),
       "comments_count": int(stt.get("commentCount", 0)),
       "duration_iso": cd.get("duration", ""),
   }

def fetch_comments(api_key, vid, cap):
    if cap == 0:
        return pd.DataFrame(columns=["comment_text", "like_count", "published_at"])
    yt = build("youtube", "v3", developerKey=api_key)
    out, got, page = [], 0, None
    while got < cap:
        try:
            r = yt.commentThreads().list(
                part="snippet",
                videoId=vid,
                maxResults=min(100, cap - got),
                pageToken=page,
                order="relevance",
                textFormat="html"
            ).execute()
        except Exception:
            break
        for it in r.get("items", []):
            top = it["snippet"]["topLevelComment"]["snippet"]
            out.append({
                "comment_text": clean_html(top.get("textDisplay", "")),
                "like_count": top.get("likeCount", 0),
                "published_at": top.get("publishedAt", "")
            })
        got += len(r.get("items", []))
        page = r.get("nextPageToken")
        if not page:
            break
        time.sleep(0.1)
    return pd.DataFrame(out)

def get_channel_videos(api_key, channel_id, max_results=30):
    yt = build("youtube", "v3", developerKey=api_key)
    videos = []
    next_page = None
    while len(videos) < max_results:
        request = yt.search().list(
            part="snippet",
            channelId=channel_id,
            maxResults=50,
            pageToken=next_page,
            type="video"
        )
        response = request.execute()
        for item in response.get("items", []):
            videos.append(item["id"]["videoId"])
        next_page = response.get("nextPageToken")
        if not next_page:
            break
        time.sleep(0.3)
    return videos

def get_channel_engagement(api_key, channel_id, max_results=30):
   videos = get_channel_videos(api_key, channel_id, max_results)
   rates = []
   for vid in videos:
       v = get_video_details(api_key, vid)
       if v:
           rate = (v["likes"] + v["comments_count"]) / max(v["views"], 1)
           rates.append(rate)
   return np.mean(rates) if rates else np.nan

# =========================================================
# EXTRA / VISUAL
# =========================================================
def generate_wordcloud(text):
   wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
   return wordcloud.to_array()

def extract_keywords_from_video(video_details):
   text = " ".join([video_details["title"]] + video_details.get("tags", []))
   vectorizer = CountVectorizer(stop_words="english", max_features=10)
   X = vectorizer.fit_transform([text])
   return vectorizer.get_feature_names_out()

def get_mid_video_thumbnail(video_id):
   return f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg"

def plot_top_comment_words(comments_df, n=15):
   words = " ".join(comments_df["comment_text"].astype(str)).lower().split()
   words = [w for w in words if w.isalpha() and w not in STOPWORDS and len(w) > 2]
   top = Counter(words).most_common(n)
   if not top:
       st.info("Not enough relevant words to display.")
       return
   df = pd.DataFrame(top, columns=["word", "count"])
   st.bar_chart(df.set_index("word"))

def plot_channel_history(api_key, channel_id, n=10):
   videos = get_channel_videos(api_key, channel_id, max_results=n)
   data = []
   for vid in videos:
       v = get_video_details(api_key, vid)
       if v:
           v["engagement_rate"] = (v["likes"] + v["comments_count"]) / max(v["views"], 1)
           data.append(v)
   if data:
       df = pd.DataFrame(data).sort_values("published_at")
       st.line_chart(df.set_index("published_at")["engagement_rate"])
   else:
       st.info("No data available for channel history.")

# =========================================================
# SIDEBAR
# =========================================================
api_key = os.getenv("YT_API_KEY") or st.sidebar.text_input("🔑 YouTube API key", type="password")

# =========================================================
# UI TABS
# =========================================================
tab1, tab2 = st.tabs(["🎥 Video Analysis", "📊 Dashboard"])

# =========================================================
# TAB 1
# =========================================================
with tab1:
   st.header("🔧 Inputs")
   c1, c2, c3 = st.columns([1, 1, 1])
   with c1:
       video_url = st.text_input("🎥 URL of the first video",
                                 placeholder="https://www.youtube.com/watch?v=VIDEO_ID_1")
   with c2:
       channel_id = st.text_input("📺 Channel ID",
                                  placeholder="e.g. UC_x5XG1OV2P6uZZ5bZ4pV6o")
   with c3:
       video_url_2 = st.text_input("🎥 URL of the second video",
                                   placeholder="https://www.youtube.com/watch?v=VIDEO_ID_2")

   max_comments = st.slider("Download comments (for sentiment analysis)", 0, 200, 50)

   colb1, colb2 = st.columns(2)
   with colb1:
       run = st.button("Analyze Main Video")
   with colb2:
       compare = st.button("Compare Engagement Rates (2 videos)")

   st.markdown("---")

   # ---------- ANALYSIS ----------
   if run:
       if not api_key:
           st.error("Please enter your API key.")
           st.stop()

       vid = extract_video_id(video_url)
       if not vid:
           st.error("Invalid video URL.")
           st.stop()

       with st.spinner("Analyzing video..."):
           v = get_video_details(api_key, vid)
           if not v:
               st.error("Could not retrieve video information.")
               st.stop()

           engagement_rate = (v["likes"] + v["comments_count"]) / max(v["views"], 1)
           st.session_state["last_analysis"] = {
               "video_id": v["video_id"],
               "title": v["title"],
               "channel": v["channel_title"],
               "engagement_rate": engagement_rate,
               "date": datetime.now().strftime("%Y-%m-%d"),
           }

           col1, col2 = st.columns([1, 2])
           with col1:
               st.image(v["thumbnail"], width=400)
           with col2:
               st.subheader(v["title"])
               st.write(f"**Channel:** {v['channel_title']}")
               st.write(f"**Published on:** {v['published_at']} ({days_since(v['published_at'])} days ago)")
               st.write(f"**Duration:** {iso8601_to_hms(v['duration_iso'])}")
               if v["tags"]:
                   st.write("**Tags:**", ", ".join(v["tags"][:10]))

           avg_eng_rate = get_channel_engagement(api_key, channel_id) if channel_id else np.nan

           st.markdown("### 📈 Channel Comparison")
           st.write(f"**Video Engagement Rate:** {engagement_rate:.4f}")
           if not np.isnan(avg_eng_rate):
               st.write(f"**Channel Average:** {avg_eng_rate:.4f}")
               if engagement_rate > avg_eng_rate:
                   st.success("✅ This video performs above the channel average.")
               else:
                   st.warning("⚠️ Below the channel average.")
           else:
               st.info("Channel ID not provided or no data to compare.")

           m1, m2, m3 = st.columns(3)
           m1.metric("👁️ Views", f"{v['views']:,}")
           m2.metric("👍 Likes", f"{v['likes']:,}")
           m3.metric("💬 Comments", f"{v['comments_count']:,}")

           comments_df = fetch_comments(api_key, vid, max_comments) if max_comments > 0 else pd.DataFrame()

           if max_comments > 0:
               st.markdown("### 🗣️ Keyword Analysis (from comments)")
               if not comments_df.empty:
                   all_comments = " ".join(comments_df["comment_text"].values)
                   st.image(generate_wordcloud(all_comments))
               else:
                   st.info("No comments available.")

               st.markdown("### 💭 Comment Sentiment & Word Trends")
               if not comments_df.empty:
                   comments_df["sentiment"] = comments_df["comment_text"].apply(
                       lambda t: sia.polarity_scores(str(t))["compound"]
                   )
                   st.write(
                       f"Comments downloaded: {len(comments_df)} | Average sentiment: {comments_df['sentiment'].mean():.3f}"
                   )
                   plot_top_comment_words(comments_df)
                   st.dataframe(comments_df.head(20))
               else:
                   st.info("No comments available for sentiment analysis.")

           # descarga CSV
           df_out = pd.DataFrame(
               [{**v, "engagement_rate": engagement_rate, "channel_avg": avg_eng_rate}]
           )
           st.download_button("📥 Download CSV",
                              df_out.to_csv(index=False).encode(),
                              "youtube_metrics.csv",
                              "text/csv")

           st.markdown("### 🔍 Video Keywords")
           st.write(", ".join(extract_keywords_from_video(v)))

           st.markdown("### 🎬 Most Watched Moment")
           st.image(get_mid_video_thumbnail(vid), caption="Thumbnail at 50% of the video")

           if channel_id:
               st.markdown("### ⏳ Engagement History")
               plot_channel_history(api_key, channel_id)

   if st.session_state["last_analysis"] is not None:
       if st.button("💾 Save this analysis"):
           st.session_state["portfolio"].append(st.session_state["last_analysis"])
           append_to_sheet(st.session_state["last_analysis"])
           st.success("Analysis saved to Dashboard (Google Sheets)!")

   # ---------- COMPARISON ----------
   if compare:
       if not api_key:
           st.error("Please enter your API key.")
           st.stop()

       vid1 = extract_video_id(video_url)
       vid2 = extract_video_id(video_url_2)

       if not vid1 or not vid2:
           st.error("❗ Invalid URL for one or both videos.")
           st.stop()

       with st.spinner("Comparing videos..."):
           v1 = get_video_details(api_key, vid1)
           v2 = get_video_details(api_key, vid2)
           if not v1 or not v2:
               st.error("Could not retrieve information for one or both videos.")
               st.stop()

           eng1 = (v1["likes"] + v1["comments_count"]) / max(v1["views"], 1)
           eng2 = (v2["likes"] + v2["comments_count"]) / max(v2["views"], 1)

           st.markdown("### 📊 Comparison Results")
           colc1, colc2 = st.columns(2)
           with colc1:
               st.subheader("Video 1")
               st.image(v1["thumbnail"], width=350)
               st.write(f"**Title:** {v1['title']}")
               st.write(f"**Engagement Rate:** {eng1:.4f}")
           with colc2:
               st.subheader("Video 2")
               st.image(v2["thumbnail"], width=350)
               st.write(f"**Title:** {v2['title']}")
               st.write(f"**Engagement Rate:** {eng2:.4f}")

           if eng1 > eng2:
               st.success("✅ Video 1 has a higher engagement rate.")
           elif eng2 > eng1:
               st.success("✅ Video 2 has a higher engagement rate.")
           else:
               st.info("➡️ Both videos have the same engagement rate.")

# =========================================================
# TAB 2 - DASHBOARD
# =========================================================
with tab2:
   st.header("📁 Saved Analyses Portfolio")

   if st.session_state["portfolio"]:
       dfp = pd.DataFrame(st.session_state["portfolio"])
       st.dataframe(dfp)

       if "engagement_rate" in dfp.columns:
           dfp["engagement_rate"] = pd.to_numeric(dfp["engagement_rate"], errors="coerce")

       colA, colB, colC = st.columns(3)

       colA.metric("Analyses saved", len(dfp))

       if "engagement_rate" in dfp.columns and dfp["engagement_rate"].notna().any():
           best_eng = dfp["engagement_rate"].max()
           colB.metric("Best engagement", f"{best_eng:.5f}")
       else:
           colB.metric("Best engagement", "—")

       if "date" in dfp.columns and dfp["date"].notna().any():
           colC.metric("Last saved", dfp["date"].iloc[-1])
       else:
           colC.metric("Last saved", "—")

       if (
           "date" in dfp.columns
           and "engagement_rate" in dfp.columns
           and dfp["engagement_rate"].notna().any()
       ):
           st.markdown("### 📈 Engagement over saved analyses")
           st.line_chart(dfp.set_index("date")["engagement_rate"])

       # top videos
       if "title" in dfp.columns and "engagement_rate" in dfp.columns:
           st.markdown("### 🏅 Top videos by engagement")
           top_df = (
               dfp.dropna(subset=["engagement_rate"])
                  .sort_values("engagement_rate", ascending=False)
                  .head(5)
           )
           if len(top_df):
               st.bar_chart(top_df.set_index("title")["engagement_rate"])
           else:
               st.info("No engagement data yet. Save an analysis first.")
   else:
       st.info("No saved analyses yet. Go to 'Video Analysis', run an analysis and click 'Save this analysis'.")
