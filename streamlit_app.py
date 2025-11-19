import os
import re
import time
import json
import io
from collections import Counter
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import requests
from dateutil import parser as dtparser
from bs4 import BeautifulSoup

import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA

# Optional libraries for thumbnail analysis
try:
    from PIL import Image, ImageStat
except Exception:
    Image = None
try:
    import cv2
except Exception:
    cv2 = None
try:
    import pytesseract
except Exception:
    pytesseract = None

# YouTube API
from googleapiclient.discovery import build

# Google Sheets
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Visualization
import altair as alt

# =========================================================
# STREAMLIT CONFIG
# =========================================================
st.set_page_config(page_title="YouTube Ad Analyzer Pro", layout="wide")
st.title("📊 YouTube Ad Analyzer – Marketing Insights Pro")
st.caption("Analyze video performance, audience sentiment, and channel history with advanced AI-powered insights.")

# =========================================================
# NLTK SETUP
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
# GOOGLE SHEETS INTEGRATION
# =========================================================
SHEET_NAME = "Marketing dashboard data"

def get_sheet():
    """Get Google Sheet using JSON stored in environment variable."""
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.file",
        "https://www.googleapis.com/auth/drive",
    ]

    raw_json = os.getenv("GOOGLE_CREDENTIALS_JSON")
    if not raw_json:
        st.error("GOOGLE_CREDENTIALS_JSON secret not found. Add it in your deployment settings.")
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

# Añadido: Control para evitar re-análisis automático
if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False

if "comparison_done" not in st.session_state:
    st.session_state["comparison_done"] = False

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def extract_video_id(url_or_id):
    """Extract video ID from YouTube URL or return ID if already valid."""
    if re.match(r"^[\w-]{11}$", url_or_id):
        return url_or_id
    m = re.search(r"v=([\w-]{11})", url_or_id) or re.search(r"youtu\.be/([\w-]{11})", url_or_id)
    return m.group(1) if m else ""

def iso8601_to_hms(iso):
    """Convert ISO 8601 duration to HH:MM:SS format."""
    h = re.search(r"(\d+)H", iso or "")
    m = re.search(r"(\d+)M", iso or "")
    s = re.search(r"(\d+)S", iso or "")
    hh, mm, ss = int(h.group(1)) if h else 0, int(m.group(1)) if m else 0, int(s.group(1)) if s else 0
    return f"{hh:02d}:{mm:02d}:{ss:02d}"

def iso8601_to_seconds(iso):
    """Convert ISO 8601 duration to seconds."""
    h = re.search(r"(\d+)H", iso or "")
    m = re.search(r"(\d+)M", iso or "")
    s = re.search(r"(\d+)S", iso or "")
    hh, mm, ss = int(h.group(1)) if h else 0, int(m.group(1)) if m else 0, int(s.group(1)) if s else 0
    return hh * 3600 + mm * 60 + ss

def clean_html(t):
    """Remove HTML tags from text."""
    return BeautifulSoup(t or "", "html.parser").get_text(" ", strip=True)

def days_since(iso_date):
    """Calculate days since a given ISO date."""
    try:
        dt = dtparser.isoparse(iso_date).astimezone(timezone.utc)
        return max((datetime.now(timezone.utc) - dt).days, 1)
    except Exception:
        return np.nan

# =========================================================
# YOUTUBE API FUNCTIONS
# =========================================================
def get_video_details(api_key, vid):
    """Fetch video details from YouTube API."""
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
        "channel_id": sn.get("channelId", ""),
        "published_at": sn.get("publishedAt", ""),
        "description": sn.get("description", ""),
        "tags": sn.get("tags", []),
        "thumbnail": sn.get("thumbnails", {}).get("high", {}).get("url"),
        "views": int(stt.get("viewCount", 0)),
        "likes": int(stt.get("likeCount", 0)),
        "comments_count": int(stt.get("commentCount", 0)),
        "duration_iso": cd.get("duration", ""),
    }

def fetch_comments(api_key, vid, cap):
    """Fetch comments from a video."""
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
    """Get list of video IDs from a channel."""
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
    """Calculate average engagement rate for a channel."""
    videos = get_channel_videos(api_key, channel_id, max_results)
    rates = []
    for vid in videos:
        v = get_video_details(api_key, vid)
        if v:
            rate = (v["likes"] + v["comments_count"]) / max(v["views"], 1)
            rates.append(rate)
    return np.mean(rates) if rates else np.nan

# =========================================================
# ADVERTISING-SPECIFIC FUNCTIONS
# =========================================================
def analyze_video_duration_for_ads(duration_seconds):
    """Analyze if video duration is optimal for advertising."""
    if duration_seconds < 6:
        return "⚡ Bumper Ad (6s) - Ideal for quick brand awareness", "success"
    elif duration_seconds <= 15:
        return "✅ Short format - Perfect for skippable ads", "success"
    elif duration_seconds <= 30:
        return "👍 Good duration - Optimal for TrueView ads", "success"
    elif duration_seconds <= 60:
        return "⚠️ Medium duration - May lose audience after 30s", "warning"
    else:
        minutes = duration_seconds // 60
        return f"❌ Too long ({minutes}min+) - High abandonment risk for ads", "error"

def analyze_title_for_ads(title):
    """Analyze title effectiveness for advertising."""
    score = 0
    recommendations = []
    
    if 40 <= len(title) <= 70:
        score += 25
        recommendations.append("✅ Optimal length (40-70 characters)")
    elif len(title) < 40:
        score += 10
        recommendations.append("⚠️ Short title - consider adding more context")
    else:
        recommendations.append("❌ Title too long - may be cut off on mobile")
    
    if re.search(r'\d+', title):
        score += 20
        recommendations.append("✅ Contains numbers (increases CTR)")
    else:
        recommendations.append("💡 Consider adding numbers for impact")
    
    power_words = ['free', 'new', 'best', 'top', 'how', 'why', 'ultimate', 'guide', 'tips']
    if any(word in title.lower() for word in power_words):
        score += 25
        recommendations.append("✅ Contains power words")
    else:
        recommendations.append("💡 Add action keywords")
    
    if title[0].isupper():
        score += 15
        recommendations.append("✅ First letter capitalized")
    
    if any(p in title for p in ['!', '?', ':', '|']):
        score += 15
        recommendations.append("✅ Uses effective punctuation")
    else:
        recommendations.append("💡 Consider punctuation for emphasis")
    
    return score, recommendations

def detect_cta_in_description(description):
    """Detect call-to-action elements in description."""
    cta_patterns = [
        r'(click|tap|visit|check out|learn more|subscribe|buy|shop|get|download)',
        r'(link in.*description|link below|in.*bio)',
        r'(www\.|https?://)',
    ]
    
    ctas_found = []
    for pattern in cta_patterns:
        matches = re.findall(pattern, description.lower())
        if matches:
            ctas_found.extend(matches)
    
    return len(set(ctas_found))

def analyze_engagement_velocity(views, likes, comments, days_old):
    """Analyze engagement velocity (daily metrics)."""
    if days_old == 0:
        days_old = 1
    
    views_per_day = views / days_old
    likes_per_day = likes / days_old
    comments_per_day = comments / days_old
    
    return {
        "views_per_day": views_per_day,
        "likes_per_day": likes_per_day,
        "comments_per_day": comments_per_day,
        "engagement_per_day": (likes + comments) / days_old
    }

def calculate_viral_coefficient(views, likes, comments, channel_avg_engagement):
    """Calculate virality coefficient compared to channel average."""
    video_engagement = (likes + comments) / max(views, 1)
    
    if channel_avg_engagement > 0:
        viral_score = (video_engagement / channel_avg_engagement) * 100
        
        if viral_score >= 150:
            return viral_score, "🚀 VIRAL - Exceeds channel average by 150%+", "success"
        elif viral_score >= 100:
            return viral_score, "📈 Excellent - Above average performance", "success"
        elif viral_score >= 75:
            return viral_score, "👍 Good - Near average performance", "info"
        else:
            return viral_score, "⚠️ Below average performance", "warning"
    
    return None, "⚠️ No channel data for comparison", "info"

def analyze_comment_sentiment_detailed(comments_df):
    """Detailed sentiment analysis of comments."""
    if comments_df.empty:
        return None
    
    comments_df["sentiment"] = comments_df["comment_text"].apply(
        lambda t: sia.polarity_scores(str(t))["compound"]
    )
    
    positive = len(comments_df[comments_df["sentiment"] > 0.05])
    neutral = len(comments_df[(comments_df["sentiment"] >= -0.05) & (comments_df["sentiment"] <= 0.05)])
    negative = len(comments_df[comments_df["sentiment"] < -0.05])
    
    total = len(comments_df)
    
    return {
        "positive_pct": (positive / total) * 100 if total > 0 else 0,
        "neutral_pct": (neutral / total) * 100 if total > 0 else 0,
        "negative_pct": (negative / total) * 100 if total > 0 else 0,
        "avg_sentiment": comments_df["sentiment"].mean(),
        "positive": positive,
        "neutral": neutral,
        "negative": negative,
        "total": total
    }

def extract_top_keywords(title, tags, description):
    """Extract top keywords from video content."""
    text = f"{title} {' '.join(tags)} {description[:200]}"
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    
    vectorizer = CountVectorizer(stop_words="english", max_features=15, ngram_range=(1, 2))
    try:
        X = vectorizer.fit_transform([text])
        keywords = vectorizer.get_feature_names_out()
        return list(keywords)
    except:
        return []

# =========================================================
# VISUALIZATION FUNCTIONS
# =========================================================
def generate_wordcloud(text):
    """Generate word cloud from text."""
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    return wordcloud.to_array()

def get_video_preview_thumbnails(video_id):
    """Get YouTube auto-generated thumbnails."""
    return {
        "beginning": f"https://img.youtube.com/vi/{video_id}/1.jpg",
        "middle": f"https://img.youtube.com/vi/{video_id}/2.jpg",
        "end": f"https://img.youtube.com/vi/{video_id}/3.jpg"
    }

def plot_top_comment_words(comments_df, n=15):
    """Plot top words from comments."""
    words = " ".join(comments_df["comment_text"].astype(str)).lower().split()
    words = [w for w in words if w.isalpha() and w not in STOPWORDS and len(w) > 2]
    top = Counter(words).most_common(n)
    if not top:
        st.info("Not enough relevant words to display.")
        return
    df = pd.DataFrame(top, columns=["word", "count"])
    st.bar_chart(df.set_index("word"))

def plot_channel_history(api_key, channel_id, n=15):
    """Plot channel engagement history."""
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
# THUMBNAIL ANALYZER (ADVANCED)
# =========================================================
def _rgb_to_hex(rgb):
    """Convert RGB tuple to hex color."""
    try:
        return '#{:02x}{:02x}{:02x}'.format(*rgb)
    except Exception:
        return None

def analyze_thumbnail(url):
    """Comprehensive thumbnail analysis."""
    out = {
        "ok": False,
        "error": None,
        "brightness": None,
        "contrast": None,
        "dominant_color_rgb": None,
        "dominant_color_hex": None,
        "ocr_text": None,
        "face_count": None,
        "size": None,
    }
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        if Image is None:
            out["error"] = "Pillow (PIL) not installed"
            return out
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        out["size"] = img.size

        gray = img.convert("L")
        stat = ImageStat.Stat(gray)
        brightness = stat.mean[0]
        contrast = stat.stddev[0]
        out["brightness"] = float(brightness)
        out["contrast"] = float(contrast)

        arr = np.array(img).reshape(-1, 3)
        arr_bucket = (arr // 32) * 32
        tuples = [tuple(int(x) for x in row) for row in arr_bucket]
        if tuples:
            dom_rgb = Counter(tuples).most_common(1)[0][0]
            out["dominant_color_rgb"] = dom_rgb
            out["dominant_color_hex"] = _rgb_to_hex(dom_rgb)

        if pytesseract is not None:
            try:
                t_cmd = os.getenv("TESSERACT_CMD")
                if t_cmd:
                    pytesseract.pytesseract.tesseract_cmd = t_cmd
                ocr_text = pytesseract.image_to_string(img)
                out["ocr_text"] = ocr_text.strip() if ocr_text and ocr_text.strip() else None
            except Exception as e:
                out["ocr_text"] = f"OCR error: {e}"

        if cv2 is not None:
            try:
                np_img = np.array(img)[:, :, ::-1].copy()
                gray_cv = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
                cascade_path = None
                if hasattr(cv2.data, "haarcascades"):
                    candidate = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
                    if os.path.exists(candidate):
                        cascade_path = candidate
                if cascade_path:
                    face_cascade = cv2.CascadeClassifier(cascade_path)
                    faces = face_cascade.detectMultiScale(gray_cv, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
                    out["face_count"] = int(len(faces))
                else:
                    out["face_count"] = None
            except Exception as e:
                out["face_count"] = f"Face detect error: {e}"

        out["ok"] = True
        return out
    except Exception as e:
        out["error"] = str(e)
        return out

# =========================================================
# EMOTION MAP & TOPIC MODELING
# =========================================================
def emotion_and_topics_from_comments(comments_df, n_topics=3, max_features=5000):
    """Advanced comment analysis."""
    res = {
        "sentiment_counts": None,
        "avg_sentiment": None,
        "repr_comments": {},
        "topics": [],
    }
    if comments_df is None or comments_df.empty:
        return res

    comments_df = comments_df.copy()
    comments_df["sentiment"] = comments_df["comment_text"].apply(
        lambda t: sia.polarity_scores(str(t))["compound"]
    )
    comments_df["sent_bucket"] = comments_df["sentiment"].apply(
        lambda s: "positive" if s > 0.2 else ("negative" if s < -0.2 else "neutral")
    )

    res["sentiment_counts"] = comments_df["sent_bucket"].value_counts().to_dict()
    res["avg_sentiment"] = float(comments_df["sentiment"].mean())

    for b in ["positive", "neutral", "negative"]:
        dfb = comments_df[comments_df["sent_bucket"] == b]
        if not dfb.empty:
            dfb_sorted = dfb.sort_values("like_count", ascending=False)
            res["repr_comments"][b] = dfb_sorted["comment_text"].head(5).tolist()
        else:
            res["repr_comments"][b] = []

    texts = comments_df["comment_text"].astype(str).tolist()
    if len(texts) >= 10:
        try:
            cv = CountVectorizer(stop_words="english", max_features=max_features)
            X = cv.fit_transform(texts)
            lda = LDA(n_components=min(n_topics, 6), random_state=0, learning_method="batch")
            lda.fit(X)
            words = cv.get_feature_names_out()
            topics = []
            for i, comp in enumerate(lda.components_):
                terms = [words[idx] for idx in comp.argsort()[-8:][::-1]]
                topics.append({"topic_id": i, "top_words": terms})
            res["topics"] = topics
        except Exception:
            res["topics"] = []
    else:
        res["topics"] = []

    return res

# =========================================================
# CHANNEL TAG HISTORY & LENGTH VS ENGAGEMENT
# =========================================================
def channel_tags_history(api_key, channel_id, max_videos=50):
    """Analyze channel tag usage and collect video data."""
    vids = get_channel_videos(api_key, channel_id, max_results=max_videos)
    tags_counter = Counter()
    data = []
    for vid in vids:
        v = get_video_details(api_key, vid)
        if v:
            data.append(v)
            tags_counter.update([t.lower() for t in v.get("tags", []) if isinstance(t, str)])
    return tags_counter, pd.DataFrame(data)

# =========================================================
# SIDEBAR
# =========================================================
api_key = os.getenv("YT_API_KEY") or st.sidebar.text_input("🔑 YouTube API Key", type="password")

# =========================================================
# UI TABS
# =========================================================
tab1, tab2 = st.tabs(["🎥 Video Analysis", "📊 Dashboard"])

# =========================================================
# TAB 1: VIDEO ANALYSIS
# =========================================================
with tab1:
    st.header("🔧 Analysis Inputs")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        video_url = st.text_input("🎥 Main Video URL",
                                  placeholder="https://www.youtube.com/watch?v=VIDEO_ID")
    with c2:
        channel_id = st.text_input("📺 Channel ID (optional)",
                                   placeholder="UC_x5XG1OV2P6uZZ5FSM9Ttw")
    with c3:
        video_url_2 = st.text_input("🎥 Comparison Video URL",
                                    placeholder="https://www.youtube.com/watch?v=VIDEO_ID_2")

    max_comments = st.slider("Comments to download (for sentiment analysis)", 0, 500, 100)

    colb1, colb2 = st.columns(2)
    with colb1:
        run = st.button("🚀 Analyze Main Video", type="primary")
    with colb2:
        compare = st.button("🆚 Compare Two Videos")

    st.markdown("---")

    # ========== MAIN VIDEO ANALYSIS ==========
    if run:
        st.session_state["analysis_done"] = True
        st.session_state["comparison_done"] = False
        
        if not api_key:
            st.error("⚠️ Please enter your YouTube API key.")
            st.stop()

        vid = extract_video_id(video_url)
        if not vid:
            st.error("❌ Invalid video URL.")
            st.stop()

        with st.spinner("🔍 Analyzing video..."):
            v = get_video_details(api_key, vid)
            if not v:
                st.error("❌ Could not retrieve video information.")
                st.stop()

            engagement_rate = (v["likes"] + v["comments_count"]) / max(v["views"], 1)
            duration_seconds = iso8601_to_seconds(v["duration_iso"])
            days_old = days_since(v["published_at"])
            
            st.session_state["last_analysis"] = {
                "video_id": v["video_id"],
                "title": v["title"],
                "channel": v["channel_title"],
                "engagement_rate": engagement_rate,
                "date": datetime.now().strftime("%Y-%m-%d"),
            }

            # ===== VIDEO HEADER =====
            col1, col2 = st.columns([1, 2])
            with col1:
                if v.get("thumbnail"):
                    st.image(v["thumbnail"], width=400)
            with col2:
                st.subheader(v["title"])
                st.write(f"**Channel:** {v['channel_title']}")
                st.write(f"**Published:** {v['published_at'][:10]} ({days_old} days ago)")
                st.write(f"**Duration:** {iso8601_to_hms(v['duration_iso'])} ({duration_seconds}s)")
                if v.get("tags"):
                    st.write(f"**Tags:** {', '.join(v['tags'][:10])}")

            # ===== CORE METRICS =====
            st.markdown("### 📊 Core Performance Metrics")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("👁️ Views", f"{v['views']:,}")
            m2.metric("👍 Likes", f"{v['likes']:,}")
            m3.metric("💬 Comments", f"{v['comments_count']:,}")
            m4.metric("📈 Engagement", f"{engagement_rate:.4f}")

            # ===== AD DURATION ANALYSIS =====
            st.markdown("### ⏱️ Ad Duration Optimization")
            duration_msg, duration_type = analyze_video_duration_for_ads(duration_seconds)
            if duration_type == "success":
                st.success(duration_msg)
            elif duration_type == "warning":
                st.warning(duration_msg)
            else:
                st.error(duration_msg)

            # ===== TITLE OPTIMIZATION =====
            st.markdown("### 📝 Title Optimization Score")
            title_score, title_recommendations = analyze_title_for_ads(v["title"])
            
            col_score, col_bar = st.columns([1, 3])
            with col_score:
                st.metric("Score", f"{title_score}/100")
            with col_bar:
                st.progress(title_score / 100)
            
            for rec in title_recommendations:
                st.write(rec)

            # ===== ENGAGEMENT VELOCITY =====
            st.markdown("### 🚀 Engagement Velocity")
            st.caption("Performance metrics normalized by days since publication")
            velocity = analyze_engagement_velocity(v["views"], v["likes"], v["comments_count"], days_old)
            
            vel1, vel2, vel3, vel4 = st.columns(4)
            vel1.metric("Views/Day", f"{velocity['views_per_day']:,.0f}")
            vel2.metric("Likes/Day", f"{velocity['likes_per_day']:,.1f}")
            vel3.metric("Comments/Day", f"{velocity['comments_per_day']:,.1f}")
            vel4.metric("Engagement/Day", f"{velocity['engagement_per_day']:,.1f}")

            # ===== CTA ANALYSIS =====
            st.markdown("### 🎯 Call-to-Action Analysis")
            cta_count = detect_cta_in_description(v.get("description", ""))
            if cta_count > 0:
                st.success(f"✅ Found {cta_count} CTA elements in description")
            else:
                st.warning("⚠️ No clear CTAs detected - consider adding links or action prompts")
            
            with st.expander("📄 View Full Description"):
                st.text(v.get("description", "No description available")[:1000] + "..." if len(v.get("description", "")) > 1000 else v.get("description", ""))

            # ===== CHANNEL BENCHMARK =====
            use_channel = channel_id if channel_id else v.get("channel_id")
            if use_channel:
                st.markdown("### 📈 Channel Performance Benchmark")
                
                with st.spinner("📊 Analyzing channel average..."):
                    avg_eng_rate = get_channel_engagement(api_key, use_channel)
                    
                    if not np.isnan(avg_eng_rate):
                        viral_score, viral_msg, viral_status = calculate_viral_coefficient(
                            v["views"], v["likes"], v["comments_count"], avg_eng_rate
                        )
                        
                        col_vid, col_chan, col_viral = st.columns(3)
                        col_vid.metric("Video Engagement", f"{engagement_rate:.4f}")
                        col_chan.metric("Channel Average", f"{avg_eng_rate:.4f}")
                        if viral_score:
                            col_viral.metric("Viral Score", f"{viral_score:.0f}%")
                        
                        if viral_status == "success":
                            st.success(viral_msg)
                        elif viral_status == "warning":
                            st.warning(viral_msg)
                        else:
                            st.info(viral_msg)
                    else:
                        st.info("⚠️ Could not calculate channel average")
            else:
                st.info("💡 Provide Channel ID for performance comparison")

            # ===== TOP KEYWORDS =====
            st.markdown("### 🔍 Top Keywords Extracted")
            keywords = extract_top_keywords(v["title"], v.get("tags", []), v.get("description", ""))
            if keywords:
                st.write(", ".join([f"`{k}`" for k in keywords[:10]]))
            else:
                st.info("No keywords extracted")

            # ===== THUMBNAIL ANALYSIS =====
            st.markdown("### 🖼️ Advanced Thumbnail Analysis")
            st.caption("AI-powered analysis of brightness, contrast, dominant colors, text (OCR), and faces")
            thumb_analysis = analyze_thumbnail(v["thumbnail"]) if v.get("thumbnail") else {"ok": False, "error": "No thumbnail"}
            
            if not thumb_analysis["ok"]:
                st.warning(f"⚠️ Thumbnail analysis unavailable: {thumb_analysis.get('error')}")
            else:
                col_t1, col_t2 = st.columns([2, 1])
                
                with col_t1:
                    st.write(f"**Size:** {thumb_analysis.get('size')}")
                    st.write(f"**Brightness (0-255):** {thumb_analysis.get('brightness'):.1f}")
                    st.write(f"**Contrast (stddev):** {thumb_analysis.get('contrast'):.1f}")
                    
                    brightness = thumb_analysis.get('brightness', 0)
                    if brightness < 80:
                        st.write("💡 **Recommendation:** Thumbnail is dark - consider brightening for better visibility")
                    elif brightness > 200:
                        st.write("💡 **Recommendation:** Thumbnail is very bright - may need more contrast")
                    else:
                        st.write("✅ **Brightness:** Optimal range for visibility")
                    
                    ocr_val = thumb_analysis.get("ocr_text")
                    if ocr_val is None:
                        st.write("**Text detected:** None")
                    elif isinstance(ocr_val, str) and ocr_val.startswith("OCR error:"):
                        st.write(f"**OCR:** {ocr_val}")
                    else:
                        st.write(f"**Text detected:** {ocr_val[:200]}")
                    
                    fc = thumb_analysis.get("face_count")
                    if fc is None:
                        st.write("**Faces detected:** OpenCV not available")
                    elif isinstance(fc, int):
                        st.write(f"**Faces detected:** {fc}")
                        if fc > 0:
                            st.write("✅ **Recommendation:** Human faces increase engagement!")
                    else:
                        st.write(f"**Faces:** {fc}")
                
                with col_t2:
                    rgb = thumb_analysis.get('dominant_color_rgb')
                    hexc = thumb_analysis.get('dominant_color_hex')
                    if rgb and hexc:
                        st.markdown("**Dominant Color**")
                        st.markdown(f"RGB: {rgb}")
                        st.markdown(f"HEX: {hexc}")
                        st.markdown(
                            f'<div style="width:100%;height:80px;border-radius:8px;border:2px solid #ddd;background:{hexc};margin-top:10px"></div>',
                            unsafe_allow_html=True
                        )

            # ===== VIDEO PREVIEW THUMBNAILS =====
            st.markdown("### 🎬 Video Preview Moments")
            st.caption("YouTube auto-generated thumbnails from different video segments")
            thumbs = get_video_preview_thumbnails(vid)
            col_thumb1, col_thumb2, col_thumb3 = st.columns(3)
            with col_thumb1:
                st.image(thumbs["beginning"], caption="Beginning", width=280)
            with col_thumb2:
                st.image(thumbs["middle"], caption="Middle", width=280)
            with col_thumb3:
                st.image(thumbs["end"], caption="End", width=280)

            # ===== COMMENT ANALYSIS =====
            comments_df = fetch_comments(api_key, vid, max_comments) if max_comments > 0 else pd.DataFrame()

            if max_comments > 0 and not comments_df.empty:
                st.markdown("### 💭 Audience Sentiment Analysis")
                sentiment_data = analyze_comment_sentiment_detailed(comments_df)
                
                if sentiment_data:
                    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                    col_s1.metric("😊 Positive", f"{sentiment_data['positive_pct']:.1f}%", 
                                 delta=f"{sentiment_data['positive']} comments")
                    col_s2.metric("😐 Neutral", f"{sentiment_data['neutral_pct']:.1f}%",
                                 delta=f"{sentiment_data['neutral']} comments")
                    col_s3.metric("😞 Negative", f"{sentiment_data['negative_pct']:.1f}%",
                                 delta=f"{sentiment_data['negative']} comments")
                    col_s4.metric("📊 Avg Score", f"{sentiment_data['avg_sentiment']:.3f}")
                    
                    if sentiment_data['positive_pct'] > 60:
                        st.success("✅ Overwhelmingly positive audience response - excellent for brand reputation")
                    elif sentiment_data['negative_pct'] > 30:
                        st.warning("⚠️ Significant negative sentiment detected - review feedback for improvements")
                    else:
                        st.info("📊 Mixed sentiment - typical for advertising content")

                st.markdown("### 🗣️ Comment Word Cloud")
                all_comments = " ".join(comments_df["comment_text"].values)
                st.image(generate_wordcloud(all_comments))

                st.markdown("### 📊 Most Frequent Words in Comments")
                plot_top_comment_words(comments_df, n=12)

                st.markdown("### 🧭 Emotion Map & Topic Detection")
                st.caption("Advanced AI analysis of audience emotions and discussion topics")
                emo = emotion_and_topics_from_comments(comments_df, n_topics=3)
                
                if emo.get("sentiment_counts"):
                    st.write(f"**Sentiment Distribution:** {emo.get('sentiment_counts')}")
                    st.write(f"**Average Sentiment Score:** {emo.get('avg_sentiment'):.3f}")
                    
                    st.markdown("**Representative Comments by Sentiment**")
                    for bucket, comments_list in emo.get("repr_comments", {}).items():
                        if comments_list:
                            with st.expander(f"💬 {bucket.title()} Comments ({len(comments_list)})"):
                                for i, comment in enumerate(comments_list[:3], 1):
                                    st.write(f"{i}. {comment}")
                    
                    if emo.get("topics"):
                        st.markdown("**Detected Discussion Topics (LDA)**")
                        for topic in emo["topics"]:
                            st.write(f"**Topic {topic['topic_id'] + 1}:** {', '.join(topic['top_words'][:6])}")
                    else:
                        st.info("💡 Not enough data for topic modeling")

                with st.expander("📝 View Sample Comments"):
                    st.dataframe(comments_df.head(20)[["comment_text", "like_count"]])

            elif max_comments > 0:
                st.info("ℹ️ No comments available for this video")

            # ===== CHANNEL TAG HISTORY & LENGTH ANALYSIS =====
            if use_channel:
                st.markdown("### 📚 Channel Deep Dive Analysis")
                st.caption("Historical tag usage and video length vs engagement patterns")
                
                with st.spinner("🔍 Analyzing channel history..."):
                    tags_counter, ch_df = channel_tags_history(api_key, use_channel, max_videos=50)
                
                if tags_counter:
                    col_tags1, col_tags2 = st.columns([1, 1])
                    
                    with col_tags1:
                        st.markdown("**Top Tags Used by Channel**")
                        top_tags = tags_counter.most_common(15)
                        df_tags = pd.DataFrame(top_tags, columns=["tag", "count"])
                        st.dataframe(df_tags)
                    
                    with col_tags2:
                        st.markdown("**Tag Frequency Distribution**")
                        st.bar_chart(df_tags.set_index("tag")["count"])
                else:
                    st.info("No tag data available")

                if not ch_df.empty:
                    ch_df["engagement_rate"] = (ch_df["likes"] + ch_df["comments_count"]) / ch_df["views"].replace(0, np.nan)
                    ch_df["duration_s"] = ch_df["duration_iso"].apply(iso8601_to_seconds)
                    
                    st.markdown("**Video Duration vs Engagement Rate**")
                    st.caption("Scatter plot showing relationship between video length and audience engagement")
                    
                    scatter_df = ch_df.dropna(subset=["duration_s", "engagement_rate"])
                    if not scatter_df.empty:
                        chart = alt.Chart(scatter_df).mark_circle(size=80, opacity=0.6).encode(
                            x=alt.X('duration_s:Q', title='Duration (seconds)', scale=alt.Scale(zero=False)),
                            y=alt.Y('engagement_rate:Q', title='Engagement Rate'),
                            color=alt.Color('engagement_rate:Q', scale=alt.Scale(scheme='viridis'), legend=None),
                            tooltip=['title:N', 'duration_s:Q', 'engagement_rate:Q', 'views:Q']
                        ).properties(
                            height=400
                        ).interactive()
                        st.altair_chart(chart, use_container_width=True)
                        
                        avg_dur = scatter_df["duration_s"].mean()
                        avg_eng = scatter_df["engagement_rate"].mean()
                        st.write(f"📊 **Channel Insights:** Average duration: {avg_dur:.0f}s ({avg_dur/60:.1f}min) | Average engagement: {avg_eng:.4f}")
                    else:
                        st.info("Not enough data for scatter plot")

                st.markdown("### ⏳ Engagement Timeline")
                plot_channel_history(api_key, use_channel, n=15)

            # ===== EXPORT DATA =====
            st.markdown("---")
            st.markdown("### 📥 Export Analysis")
            df_out = pd.DataFrame([{
                **v,
                "engagement_rate": engagement_rate,
                "duration_seconds": duration_seconds,
                "days_since_publish": days_old,
                "views_per_day": velocity['views_per_day'],
                "title_score": title_score,
                "cta_count": cta_count
            }])
            st.download_button(
                "📥 Download Full Analysis (CSV)",
                df_out.to_csv(index=False).encode(),
                "youtube_ad_analysis.csv",
                "text/csv"
            )

    # ===== SAVE ANALYSIS =====
    if st.session_state["last_analysis"] is not None and st.session_state.get("analysis_done"):
        if st.button("💾 Save Analysis to Dashboard"):
            st.session_state["portfolio"].append(st.session_state["last_analysis"])
            try:
                append_to_sheet(st.session_state["last_analysis"])
                st.success("✅ Analysis saved to Dashboard and Google Sheets!")
            except Exception as e:
                st.warning(f"⚠️ Saved locally but Google Sheets sync failed: {e}")

    # ========== VIDEO COMPARISON ==========
    if compare:
        st.session_state["comparison_done"] = True
        st.session_state["analysis_done"] = False
        
        if not api_key:
            st.error("⚠️ Please enter your YouTube API key.")
            st.stop()

        vid1 = extract_video_id(video_url)
        vid2 = extract_video_id(video_url_2)

        if not vid1 or not vid2:
            st.error("❌ Invalid URL for one or both videos.")
            st.stop()

        with st.spinner("🔍 Comparing videos..."):
            v1 = get_video_details(api_key, vid1)
            v2 = get_video_details(api_key, vid2)
            if not v1 or not v2:
                st.error("❌ Could not retrieve video information.")
                st.stop()

            eng1 = (v1["likes"] + v1["comments_count"]) / max(v1["views"], 1)
            eng2 = (v2["likes"] + v2["comments_count"]) / max(v2["views"], 1)
            
            dur1 = iso8601_to_seconds(v1["duration_iso"])
            dur2 = iso8601_to_seconds(v2["duration_iso"])
            
            days1 = days_since(v1["published_at"])
            days2 = days_since(v2["published_at"])
            
            vel1 = analyze_engagement_velocity(v1["views"], v1["likes"], v1["comments_count"], days1)
            vel2 = analyze_engagement_velocity(v2["views"], v2["likes"], v2["comments_count"], days2)

            st.markdown("### 🆚 Head-to-Head Video Comparison")
            
            colc1, colc2 = st.columns(2)
            
            with colc1:
                st.subheader("📹 Video 1")
                if v1.get("thumbnail"):
                    st.image(v1["thumbnail"], width=380)
                st.write(f"**{v1['title']}**")
                st.write(f"Channel: {v1['channel_title']}")
                st.write(f"Duration: {iso8601_to_hms(v1['duration_iso'])}")
                st.write(f"Published: {days1} days ago")
                
                st.metric("Views", f"{v1['views']:,}")
                st.metric("Likes", f"{v1['likes']:,}")
                st.metric("Comments", f"{v1['comments_count']:,}")
                st.metric("Engagement Rate", f"{eng1:.4f}")
                st.metric("Views/Day", f"{vel1['views_per_day']:,.0f}")
                
            with colc2:
                st.subheader("📹 Video 2")
                if v2.get("thumbnail"):
                    st.image(v2["thumbnail"], width=380)
                st.write(f"**{v2['title']}**")
                st.write(f"Channel: {v2['channel_title']}")
                st.write(f"Duration: {iso8601_to_hms(v2['duration_iso'])}")
                st.write(f"Published: {days2} days ago")
                
                st.metric("Views", f"{v2['views']:,}", 
                         delta=f"{((v2['views']/v1['views']-1)*100):+.1f}%" if v1['views'] > 0 else None)
                st.metric("Likes", f"{v2['likes']:,}", 
                         delta=f"{((v2['likes']/v1['likes']-1)*100):+.1f}%" if v1['likes'] > 0 else None)
                st.metric("Comments", f"{v2['comments_count']:,}", 
                         delta=f"{((v2['comments_count']/v1['comments_count']-1)*100):+.1f}%" if v1['comments_count'] > 0 else None)
                st.metric("Engagement Rate", f"{eng2:.4f}", 
                         delta=f"{((eng2/eng1-1)*100):+.1f}%" if eng1 > 0 else None)
                st.metric("Views/Day", f"{vel2['views_per_day']:,.0f}", 
                         delta=f"{((vel2['views_per_day']/vel1['views_per_day']-1)*100):+.1f}%" if vel1['views_per_day'] > 0 else None)

            st.markdown("### 🏆 Performance Winner")
            
            points_v1 = 0
            points_v2 = 0
            
            comparison_results = []
            
            if eng1 > eng2:
                points_v1 += 1
                comparison_results.append("✅ **Video 1** has higher engagement rate")
            elif eng2 > eng1:
                points_v2 += 1
                comparison_results.append("✅ **Video 2** has higher engagement rate")
            else:
                comparison_results.append("➡️ Equal engagement rate")
            
            if vel1['views_per_day'] > vel2['views_per_day']:
                points_v1 += 1
                comparison_results.append("✅ **Video 1** has better view velocity")
            elif vel2['views_per_day'] > vel1['views_per_day']:
                points_v2 += 1
                comparison_results.append("✅ **Video 2** has better view velocity")
            
            if dur1 <= 30 and dur2 > 30:
                points_v1 += 1
                comparison_results.append("✅ **Video 1** has better ad-optimized duration")
            elif dur2 <= 30 and dur1 > 30:
                points_v2 += 1
                comparison_results.append("✅ **Video 2** has better ad-optimized duration")
            
            if v1['likes']/max(v1['views'], 1) > v2['likes']/max(v2['views'], 1):
                points_v1 += 1
                comparison_results.append("✅ **Video 1** has better like ratio")
            elif v2['likes']/max(v2['views'], 1) > v1['likes']/max(v1['views'], 1):
                points_v2 += 1
                comparison_results.append("✅ **Video 2** has better like ratio")
            
            for result in comparison_results:
                st.write(result)
            
            st.markdown("---")
            if points_v1 > points_v2:
                st.success(f"🏆 **Video 1 WINS** ({points_v1} vs {points_v2} points)")
            elif points_v2 > points_v1:
                st.success(f"🏆 **Video 2 WINS** ({points_v2} vs {points_v1} points)")
            else:
                st.info(f"🤝 **TIE** ({points_v1} vs {points_v2} points)")
            
            st.markdown("### 📥 Export Comparison")
            comparison_df = pd.DataFrame([
                {
                    "video": "Video 1", 
                    "title": v1["title"], 
                    "views": v1["views"], 
                    "likes": v1["likes"],
                    "comments": v1["comments_count"],
                    "engagement": eng1, 
                    "views_per_day": vel1['views_per_day'],
                    "duration_s": dur1
                },
                {
                    "video": "Video 2", 
                    "title": v2["title"], 
                    "views": v2["views"], 
                    "likes": v2["likes"],
                    "comments": v2["comments_count"],
                    "engagement": eng2, 
                    "views_per_day": vel2['views_per_day'],
                    "duration_s": dur2
                }
            ])
            
            st.download_button(
                "📥 Download Comparison (CSV)",
                comparison_df.to_csv(index=False).encode(),
                "video_comparison.csv",
                "text/csv"
            )

# =========================================================
# TAB 2: DASHBOARD
# =========================================================
with tab2:
    st.header("📁 Saved Analyses Portfolio")
    st.caption("Track and compare all your analyzed videos over time")

    if st.session_state["portfolio"]:
        dfp = pd.DataFrame(st.session_state["portfolio"])
        
        if "engagement_rate" in dfp.columns:
            dfp["engagement_rate"] = pd.to_numeric(dfp["engagement_rate"], errors="coerce")

        st.markdown("### 📊 Portfolio Summary")
        colA, colB, colC = st.columns(3)

        colA.metric("Total Analyses", len(dfp))

        if "engagement_rate" in dfp.columns and dfp["engagement_rate"].notna().any():
            best_eng = dfp["engagement_rate"].max()
            avg_eng = dfp["engagement_rate"].mean()
            colB.metric("Best Engagement", f"{best_eng:.5f}", delta=f"Avg: {avg_eng:.5f}")
        else:
            colB.metric("Best Engagement", "—")

        if "date" in dfp.columns and dfp["date"].notna().any():
            colC.metric("Last Saved", dfp["date"].iloc[-1])
        else:
            colC.metric("Last Saved", "—")

        st.markdown("### 📋 All Saved Analyses")
        st.dataframe(dfp, use_container_width=True)

        if (
            "date" in dfp.columns
            and "engagement_rate" in dfp.columns
            and dfp["engagement_rate"].notna().any()
        ):
            st.markdown("### 📈 Engagement Timeline")
            st.caption("Track engagement rate trends across all analyzed videos")
            st.line_chart(dfp.set_index("date")["engagement_rate"])

        if "title" in dfp.columns and "engagement_rate" in dfp.columns:
            st.markdown("### 🏅 Top Performing Videos")
            top_df = (
                dfp.dropna(subset=["engagement_rate"])
                .sort_values("engagement_rate", ascending=False)
                .head(5)
            )
            if len(top_df):
                st.bar_chart(top_df.set_index("title")["engagement_rate"])
            else:
                st.info("No engagement data yet. Save an analysis first.")
        
        st.markdown("### 📥 Export Portfolio")
        st.download_button(
            "📥 Download Full Portfolio (CSV)",
            dfp.to_csv(index=False).encode(),
            "portfolio_full.csv",
            "text/csv"
        )
    else:
        st.info("📭 No saved analyses yet. Go to 'Video Analysis', run an analysis, and click 'Save Analysis to Dashboard'.")

