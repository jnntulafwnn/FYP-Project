import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
import pandas as pd
import re
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import os
import json
from io import StringIO
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import joblib

# --- CONFIGURATION ---
DATA_FILE = 'receipts_log.csv' 

# --- IMPORTANT: Set Tesseract path if needed ---
# If you get a TesseractNotFoundError, uncomment the line below and set the correct path
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# ---------------------------------------------------------
# 🎨 Page Configuration 
# ---------------------------------------------------------
st.set_page_config(
    page_title="Purchase Performance Analytics System",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 💅 Custom CSS 
# ---------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 💾 Data Persistence Functions 
# ---------------------------------------------------------

def load_receipts_data():
    """Loads receipt data from a CSV file."""
    if not os.path.exists(DATA_FILE):
        return []
    try:
        df = pd.read_csv(DATA_FILE)
        # Convert the 'items' string column back into a list of dictionaries
        df['items'] = df['items'].apply(json.loads)
        return df.to_dict('records')
    except Exception as e:
        return []

def save_receipts_data(data):
    """Saves receipt data to a CSV file."""
    if not data:
        df = pd.DataFrame(columns=['merchant_name', 'date', 'items', 'grand_total'])
    else:
        df = pd.DataFrame(data)
    
    # Convert the 'items' list of dictionaries column into a storable string format
    df['items'] = df['items'].apply(json.dumps)
    
    try:
        df.to_csv(DATA_FILE, index=False)
    except Exception as e:
        st.error(f"Error saving data: {e}")

# ---------------------------------------------------------
# 🧱 Initialize Session State 
# ---------------------------------------------------------
if 'receipts_data' not in st.session_state:
    st.session_state.receipts_data = load_receipts_data() 
if 'item_database' not in st.session_state:
    st.session_state.item_database = None
if 'manual_items' not in st.session_state:
    st.session_state.manual_items = [{'item_name': '', 'quantity': 1, 'unit_price': 0.0, 'total_price': 0.0}]
if 'temp_extracted_receipt' not in st.session_state:
    st.session_state.temp_extracted_receipt = None 
if 'processed_img_bytes' not in st.session_state:
    st.session_state.processed_img_bytes = None 
if 'raw_ocr_text' not in st.session_state:
    st.session_state.raw_ocr_text = "" 
if 'current_uploaded_file' not in st.session_state:
    st.session_state.current_uploaded_file = None # KEY: New state for persistent file object
if 'manual_total' not in st.session_state:
    st.session_state.manual_total = 0.0

# ---------------------------------------------------------
# 📘 Load Item Database 
# ---------------------------------------------------------
@st.cache_data
def load_item_database():
    try:
        # Default fallback data
        df = pd.DataFrame({
            'Item Name': ['Beras Faiza (10kg)', 'Minyak Masak Seri (5kg)', 'Gula Putih CSR (1kg)', 'Tepung Gandum Cap Sauh (1kg)', 'Susu Pekat Manis F&N (390g)', 'Pewangi Baju Downy (900ml)'],
            'Category': ['Cooking Essentials', 'Cooking Essentials', 'Cooking Essentials', 'Cooking Essentials', 'Dairy', 'Household'],
        })
        # If item_database.xlsx exists, try to load it
        if os.path.exists('item_database.xlsx'):
            df_from_file = pd.read_excel('item_database.xlsx')
            df = pd.concat([df_from_file, df]).drop_duplicates(subset=['Item Name']).reset_index(drop=True)

        return df
    except Exception:
        # Default fallback data if the file is missing or corrupted
        return pd.DataFrame({
            'Item Name': ['Beras Faiza (10kg)', 'Minyak Masak Seri (5kg)', 'Gula Putih CSR (1kg)', 'Tepung Gandum Cap Sauh (1kg)', 'Susu Pekat Manis F&N (390g)', 'Pewangi Baju Downy (900ml)'],
            'Category': ['Cooking Essentials', 'Cooking Essentials', 'Cooking Essentials', 'Cooking Essentials', 'Dairy', 'Household'],
        })


# ---------------------------------------------------------
# 🛠️ Helper Functions (OCR Resilience, Matching, Parsing)
# ---------------------------------------------------------

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=20)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )
    thresh = cv2.bitwise_not(thresh)
    
    try:
        osd = pytesseract.image_to_osd(thresh)
        angle = int(re.search(r"Rotate: (\d+)", osd).group(1))
        if angle != 0:
            (h, w) = thresh.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), -angle, 1.0)
            thresh = cv2.warpAffine(thresh, M, (w, h),
                                    flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_REPLICATE)
    except:
        pass
    return thresh

def perform_ocr(image):
    custom_config = r'--oem 3 --psm 4'
    text = pytesseract.image_to_string(image, config=custom_config, lang='eng')
    return text

# === 🧠 MACHINE LEARNING ENHANCEMENT (ROBUST, OCR-TOLERANT) ===
# Trains on the full item_database.xlsx (~300 rows),
# matches OCR strings to the closest item name in the DB,
# and returns the DB category. Falls back to ML category if no close match.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import hashlib

# --- Text normalizer to absorb common OCR glitches (e.g., 'kq' → 'kg') ---
def _normalize_item_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    t = s.lower().strip()

    # fix common OCR mistakes
    # units & letters
    t = t.replace(" kq", " kg").replace("(kq", "(kg").replace("kq)", "kg)")
    t = t.replace(" iq", " 1g")  # occasional 1g instead of 1g/1kg glitches
    t = t.replace("1q", "1g")
    t = t.replace("5q", "5g")
    t = t.replace("|", "l")     # pipe mistaken for L
    t = t.replace("0l", "0l")   # keep '0l' as '0l'
    t = t.replace("®", "r")
    t = t.replace("—", "-")

    # punctuation/spacing simplifications
    t = re.sub(r"\s{2,}", " ", t)
    t = re.sub(r"\s*-\s*", " - ", t)

    # normalize units with optional space: 10kg / 10 kg -> 10kg
    t = re.sub(r"(\d+)\s*(kg|g|ml|l)\b", r"\1\2", t)

    return t

@st.cache_resource
def train_or_load_ml_model(item_database_path: str = "item_database.xlsx"):
    """
    Train or load a Naive Bayes model for item category classification using the
    full Excel database. Uses char n-grams to be resilient to OCR typos.
    Also returns the dataframe so the matcher can snap to exact DB item names.
    """
    model_path = "ml_item_classifier.pkl"
    vectorizer_path = "ml_tfidf.pkl"
    label_encoder_path = "ml_label_encoder.pkl"
    db_cache_path = "ml_item_db.pkl"   # stores the cleaned DB used for training

    # --- Load DB (must exist) ---
    if not os.path.exists(item_database_path):
        st.error(f"❌ Missing {item_database_path}. Place it next to the app.")
        return None, None, None, None

    try:
        item_db = pd.read_excel(item_database_path)
    except Exception as e:
        st.error(f"❌ Failed reading {item_database_path}: {e}")
        return None, None, None, None

    # keep only valid rows
    item_db = item_db.dropna(subset=["Item Name", "Category"]).copy()
    item_db["Item Name"] = item_db["Item Name"].astype(str)
    item_db["Category"] = item_db["Category"].astype(str)

    # create normalized text for robust training/matching
    item_db["__norm_name__"] = item_db["Item Name"].apply(_normalize_item_text)

    # use a simple hash of the DB to invalidate stale models automatically
    db_hash = hashlib.md5(("|".join(item_db["__norm_name__"] + "§" + item_db["Category"])).encode("utf-8")).hexdigest()

    # try loading cached models that match the current DB hash
    try:
        if all(os.path.exists(p) for p in [model_path, vectorizer_path, label_encoder_path, db_cache_path]):
            saved = joblib.load(db_cache_path)
            if saved.get("db_hash") == db_hash:
                clf = joblib.load(model_path)
                tfidf = joblib.load(vectorizer_path)
                le = joblib.load(label_encoder_path)
                return clf, tfidf, le, saved["item_db"]
    except Exception as e:
        st.warning(f"⚠️ Could not load saved ML model. Retraining… ({e})")

    # --- Train fresh model on the full database ---
    if item_db.empty:
        st.warning("⚠️ Item database is empty; cannot train model.")
        return None, None, None, None

    X = item_db["__norm_name__"]
    y = item_db["Category"]

    # char n-grams are robust to OCR typos and minor spelling errors
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    X_vec = tfidf.fit_transform(X)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    clf = MultinomialNB()
    clf.fit(X_vec, y_enc)

    # save artifacts and a copy of the working DB (with hash)
    joblib.dump(clf, model_path)
    joblib.dump(tfidf, vectorizer_path)
    joblib.dump(le, label_encoder_path)
    joblib.dump({"item_db": item_db, "db_hash": db_hash}, db_cache_path)

    st.success(f"✅ Trained ML model on {len(item_db)} items across {len(le.classes_)} categories.")
    return clf, tfidf, le, item_db

# Load/train once (uses the real Excel DB)
clf_model, tfidf_vectorizer, label_encoder, _TRAIN_DB = train_or_load_ml_model("item_database.xlsx")


def match_item_with_database(item_text: str, database: pd.DataFrame):
    """
    1) Try to SNAP the OCR string to the closest item in item_database.xlsx
       using TF-IDF cosine similarity on normalized strings.
       If a good match is found, return that exact DB item name + its category.
    2) Otherwise, fall back to ML category prediction on the OCR text.
    """
    txt_norm = _normalize_item_text(item_text)

    # --- Prefer a direct match to the DB so names/categories follow Excel exactly ---
    if _TRAIN_DB is not None and tfidf_vectorizer is not None:
        try:
            # vector for OCR item
            v_q = tfidf_vectorizer.transform([txt_norm])

            # vectors for DB names (compute on the fly to stay in sync with Excel)
            db_norm_names = _TRAIN_DB["__norm_name__"].tolist()
            v_db = tfidf_vectorizer.transform(db_norm_names)

            sims = cosine_similarity(v_q, v_db).flatten()
            best_i = int(sims.argmax())
            best_score = float(sims[best_i])

            # threshold tuned for OCR noise; 0.45 works well for small typos like 'kq'->'kg'
            if best_score >= 0.45:
                matched_row = _TRAIN_DB.iloc[best_i]
                return {
                    "item_name": matched_row["Item Name"],  # exact from Excel
                    "category": matched_row["Category"],     # exact from Excel
                    "match_score": best_score,
                }
        except Exception:
            pass  # if anything fails, use ML fallback below

    # --- ML fallback: predict category from the raw OCR text ---
    if clf_model is not None and tfidf_vectorizer is not None and label_encoder is not None:
        try:
            v = tfidf_vectorizer.transform([txt_norm])
            y_pred = clf_model.predict(v)
            cat = label_encoder.inverse_transform(y_pred)[0]
            return {
                "item_name": item_text,     # keep original if we couldn't snap to Excel name
                "category": cat,
                "match_score": 0.0,
            }
        except Exception:
            pass

    # ultimate fallback
    return {"item_name": item_text, "category": "Uncategorized", "match_score": 0.0}
# === 🧠 MACHINE LEARNING ENHANCEMENT END ===



def normalize_number(s):
    if s is None or s == "":
        return None
    s = str(s).strip()
    
    s = s.replace('S', '5').replace('$', '5').replace('|', '1').replace('l', '1')
    s = re.sub(r'[^\d,.\-]', '', s) 
    
    if ',' in s and '.' in s:
        s = s.replace(',', '')
    elif ',' in s and re.search(r',\d{1,2}$', s):
        s = s.replace(',', '.')
    elif ',' in s:
        s = s.replace(',', '')
    
    try:
        f_num = float(s)
    except:
        return None
        
    if '.' not in s and f_num >= 1000:
        return f_num / 100.0
        
    return f_num

def parse_date_from_text(text):
    m_specific = re.search(r'Payment Date:[\s:]*(\d{2}/\d{2}/\d{4}|\d{2}/\d{2}/\d{2})', text)
    if m_specific:
        s = m_specific.group(1)
    else:
        m = re.search(r'(\d{1,4}[/-]\d{1,2}[/-]\d{1,4})', text)
        if not m:
            return ''
        s = m.group(1)
        
    for fmt in ('%d/%m/%Y','%d-%m-%Y','%d/%m/%y','%d-%m-%y','%Y-%m-%d','%Y/%m/%d'):
        try:
            if fmt.endswith('%y') and len(s) == 8: 
                 if int(s.split('/')[-1]) < 50: 
                     s = s[:-2] + '20' + s[-2:] 
            
            dt = datetime.strptime(s, fmt)
            return dt.strftime('%Y-%m-%d')
        except:
            continue
    return s

def extract_receipt_data(text, item_database=None):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    receipt_data = {'merchant_name': '', 'date': '', 'items': [], 'grand_total': 0.0}
    
    receipt_data['date'] = parse_date_from_text(text)
    
    merchant = ''
    for l in lines[0:5]:
        if re.search(r'PAYMENT RECEIPT|Bill To|Receipt No|Description', l, re.IGNORECASE):
            continue
        if re.search(r'[A-Za-z]{3,}', l) and len(l) > 5:
            merchant = l.replace('—', ' ').replace('Mur', 'Murni').strip() 
            break
    receipt_data['merchant_name'] = merchant

    item_pattern = re.compile(
        r'(.+?)\s+' 
        r'(\d+)\s+'
        r'(?:RM)?\s*([\d\.,]+)\s+'
        r'(?:RM)?\s*([\d\.,]+)$',
        re.MULTILINE
    )
    items = []
    
    text_to_search = text
    item_block_match = re.search(r"(Description.*Total \(RM\))(.*?)(Sub Total)", text, re.DOTALL)
    if item_block_match:
        text_to_search = item_block_match.group(2)
        
    for match in item_pattern.finditer(text_to_search):
        name, qty_s, unit_s, total_s = match.groups()
        
        name = re.sub(r'^\s*[\d\s]+|^\s*[a-z]\s+', '', name, flags=re.IGNORECASE).strip()
        
        try:
            qty = int(qty_s)
        except:
            continue
            
        unit = normalize_number(unit_s)
        total = normalize_number(total_s)
        
        if total is None or qty <= 0:
            continue
        
        if 'description' in name.lower() or 'tax' in name.lower() or 'total' in name.lower() or 'sub' in name.lower():
            continue
            
        matched = match_item_with_database(name, item_database)
        items.append({
            'item_name': matched['item_name'],
            'category': matched['category'],
            'quantity': qty,
            'unit_price': unit,
            'total_price': total,
            'match_score': matched['match_score']
        })
    receipt_data['items'] = items
    
    grand_total = None
    for l in reversed(lines):
        m = re.search(r'(Amount Paid|Total Paid|TOTAL \(RM\)|TOTAL:|GRAND TOTAL|Sub Total)\s*[:\-]*\s*([\d\.,\$S]+)', l, re.IGNORECASE)
        if m:
            num = normalize_number(m.group(2))
            if num is not None and num > 1.0: 
                grand_total = num
                break
    
    if grand_total is None and items:
        grand_total = sum(i['total_price'] for i in items)
        
    receipt_data['grand_total'] = grand_total if grand_total else 0.0
    return receipt_data

# ---------------------------------------------------------
# 📊 Chart Functions 
# ---------------------------------------------------------
def create_category_distribution(data):
    category_totals = {}
    for receipt in data:
        for item in receipt['items']:
            category = item['category']
            category_totals[category] = category_totals.get(category, 0) + item['total_price']
    df = pd.DataFrame(list(category_totals.items()), columns=['Category', 'Total'])
    if df.empty:
        return px.pie(title='Spending by Category - No Data')
    return px.pie(df, values='Total', names='Category', title='Spending by Category')

def create_spending_timeline(data):
    dates, totals = [], []
    for receipt in data:
        if receipt['date']:
            dates.append(receipt['date'])
            totals.append(receipt['grand_total'])
    df = pd.DataFrame({'Date': dates, 'Total': totals})
    if df.empty:
        return px.line(title='Spending Over Time - No Data')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df_grouped = df.groupby('Date')['Total'].sum().reset_index()
    return px.line(df_grouped, x='Date', y='Total', title='Spending Over Time', markers=True)

def create_top_items_chart(data):
    item_counts = {}
    for receipt in data:
        for item in receipt['items']:
            item_counts[item['item_name']] = item_counts.get(item['item_name'], 0) + item['quantity']
    top_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    df = pd.DataFrame(top_items, columns=['Item', 'Quantity'])
    if df.empty:
        return px.bar(title='Top 10 Purchased Items - No Data')
    return px.bar(df, x='Quantity', y='Item', orientation='h', title='Top 10 Purchased Items (By Quantity)')

def create_receipts_by_merchant_chart(data): 
    merchant_counts = {}
    for receipt in data:
        merchant = receipt['merchant_name'] if receipt['merchant_name'] else "Unknown Merchant"
        merchant_counts[merchant] = merchant_counts.get(merchant, 0) + 1
    
    df = pd.DataFrame(list(merchant_counts.items()), columns=['Merchant', 'Receipt Count'])
    if df.empty:
        return px.bar(title='Number of Receipts by Merchant - No Data')
    df = df.sort_values('Receipt Count', ascending=False)
    
    return px.bar(
        df, 
        x='Merchant', 
        y='Receipt Count', 
        title='Number of Receipts by Merchant',
        color='Receipt Count'
    )

def generate_ai_predictions(data):
    """Generate more human-like AI insights and predictions based on purchase history."""
    if not data:
        return "🤖 No data available for AI insights yet. Upload or enter receipts to start analysis."

    total_receipts = len(data)
    total_spending = sum(r['grand_total'] for r in data)
    avg_spending = total_spending / total_receipts if total_receipts else 0

    # Calculate per-category spending
    category_spending = {}
    for receipt in data:
        for item in receipt['items']:
            category_spending[item['category']] = category_spending.get(item['category'], 0) + item['total_price']

    # Top category
    top_category = max(category_spending.items(), key=lambda x: x[1])[0] if category_spending else "N/A"

    # Spending trend by date
    df = pd.DataFrame([(r['date'], r['grand_total']) for r in data if r.get('date')], columns=['Date', 'Total'])
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values('Date').dropna()
    df['Month'] = df['Date'].dt.to_period('M')
    monthly = df.groupby('Month')['Total'].sum().reset_index()

    # Detect trend direction
    trend_text = ""
    if len(monthly) >= 2:
        diff = monthly['Total'].iloc[-1] - monthly['Total'].iloc[-2]
        change = (diff / monthly['Total'].iloc[-2]) * 100 if monthly['Total'].iloc[-2] > 0 else 0
        if change > 10:
            trend_text = f"📈 Your spending increased by **{change:.1f}%** compared to last month."
        elif change < -10:
            trend_text = f"📉 Great job! Your spending decreased by **{abs(change):.1f}%** since last month."
        else:
            trend_text = f"⚖️ Your spending is stable compared to last month."
    else:
        trend_text = "📊 Not enough data yet to identify monthly trends."

    # Forecast based on daily average
    if not df.empty:
        days_span = (df['Date'].max() - df['Date'].min()).days or 1
        daily_avg = total_spending / days_span
        forecast = daily_avg * 30
    else:
        forecast = avg_spending * 4

    # Smart Recommendations
    recs = []
    if top_category.lower() in ["household", "groceries", "cooking essentials"]:
        recs.append("🧺 You’re spending most on household and groceries — consider bulk-buying essentials for discounts.")
    elif top_category.lower() in ["dairy", "beverages"]:
        recs.append("🥛 High dairy/beverage spending detected — try comparing prices or choosing local brands.")
    elif top_category.lower() in ["electronics", "gadgets"]:
        recs.append("💻 Electronics purchases stand out — ensure warranty tracking and avoid impulse tech buys.")
    else:
        recs.append("💡 Spending is diversified across categories, showing a balanced purchase pattern.")

    if avg_spending > 150:
        recs.append("💸 Your average receipt value is quite high — setting a purchase threshold alert could help.")
    elif avg_spending < 50:
        recs.append("🪙 Excellent cost control! Keep tracking to maintain your low purchase averages.")

    if forecast > total_spending / 1.5:
        recs.append("📆 Forecast indicates an upward trend — consider adjusting your budget upward slightly next month.")
    else:
        recs.append("📆 Forecast suggests consistent spending — you’re maintaining good control.")

    # Construct the AI summary text
    insight_text = f"""
    ### 🤖 AI-Powered Spending Insights
    Here's what I found from your receipts so far:

    **📊 Quick Summary**
    - Total Receipts: **{total_receipts}**
    - Total Spending: **RM {total_spending:.2f}**
    - Average Receipt: **RM {avg_spending:.2f}**
    - Top Category: **{top_category}**
    - Estimated Next Month Spending: **RM {forecast:.2f}**

    **📈 Trend Analysis**
    {trend_text}

    **🧠 Smart Recommendations**
    {chr(10).join(recs)}

    ---
    🪄 *Insight generated automatically using pattern analysis and purchase forecasting logic.*
    """

    return insight_text


# ---------------------------------------------------------
# 💻 Manual Input Helper Functions
# ---------------------------------------------------------

# ---------------------------------------------------------
# 📝 Manual Input Page Function
# ---------------------------------------------------------
def manual_input_page():
    st.markdown('<div class="sub-header">📝 Manual Receipt Entry</div>', unsafe_allow_html=True)
    
    st.subheader("Item Details (Auto-Calculated)")

    df_manual = pd.DataFrame(st.session_state.manual_items)
    if df_manual.empty:
        df_manual = pd.DataFrame([{'item_name': '', 'quantity': 1, 'unit_price': 0.0, 'total_price': 0.0}])

    try:
        calculated_grand_total = df_manual['total_price'].sum()
    except:
        calculated_grand_total = 0.0

    st.caption("Edit **Quantity** or **Unit Price (RM)** to automatically update the **Total Price (RM)**.")

    # --- Editable Table ---
    edited_items = st.data_editor(
        st.session_state.manual_items,
        column_config={
            "item_name": st.column_config.TextColumn("Item Name", required=True),
            "quantity": st.column_config.NumberColumn("Quantity", min_value=1, format="%d"),
            "unit_price": st.column_config.NumberColumn("Unit Price (RM)", min_value=0.01, format="%.2f"),
            "total_price": st.column_config.NumberColumn("Total Price (RM)", disabled=True, format="%.2f"),
        },
        num_rows="dynamic",
        use_container_width=True,
        key="manual_editor"
    )

    # --- AUTO-CALCULATE IMMEDIATELY HERE ---

    # --- Initialise grand total in session state (first time only) ---
    # Ensure manual_total exists
    if "manual_total" not in st.session_state:
        st.session_state.manual_total = 0.0

    # --- REFRESH BUTTON (always visible) ---
    if st.button("🔄 Refresh Totals"):
        df = pd.DataFrame(edited_items)
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0).astype(int)
        df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce").fillna(0.0)
        df["total_price"] = df["quantity"] * df["unit_price"]

        st.session_state.manual_items = df.to_dict("records")
        st.session_state.manual_total = float(df["total_price"].sum())
        st.rerun()



    st.markdown("---")
    
    with st.form("manual_receipt_form"):
        # The calculate_item_totals runs on change, updating st.session_state.manual_items
        # We need to re-read the calculated total just before the form is submitted
        # For simplicity, we calculate the total inside the form scope or trust the on_change
        
        try:
            current_df_for_total = pd.DataFrame(st.session_state.manual_items)
            current_calculated_grand_total = current_df_for_total['total_price'].sum()
        except:
            current_calculated_grand_total = 0.0
            
        col1, col2 = st.columns(2)
        with col1:
            merchant_name = st.text_input("Merchant Name", key="manual_merchant")
        with col2:
            receipt_date = st.date_input("Date", key="manual_date", value=datetime.today().date())
        
        grand_total = st.number_input(
            "Grand Total (RM) - Adjust for Discounts/Taxes if needed", 
            min_value=0.00, 
            format="%.2f", 
            value=st.session_state.manual_total, 
            key="manual_total"
        )


        submitted = st.form_submit_button("💾 Save Receipt")

        if submitted:
            final_items = []
            valid_submission = True

            # Use the data that was potentially updated by the last on_change
            df_final = pd.DataFrame(st.session_state.manual_items)
            final_calculated_total = df_final['total_price'].sum()

            for index, row in df_final.iterrows():
                if pd.isna(row['item_name']) or str(row['item_name']).strip() == '':
                    continue 

                try:
                    item_name = str(row['item_name']).strip()
                    qty = int(row['quantity'])
                    unit_price = float(row['unit_price'])
                    total_price = float(row['total_price'])
                except (ValueError, TypeError):
                    continue

                if qty <= 0 or total_price <= 0:
                    continue 

                matched = match_item_with_database(item_name, st.session_state.item_database)
                
                final_items.append({
                    'item_name': item_name,
                    'category': matched.get('category', 'Uncategorized'),
                    'quantity': qty,
                    'unit_price': unit_price,
                    'total_price': total_price,
                    'match_score': 1.0 
                })

            if not merchant_name:
                st.error("Merchant Name is required.")
                valid_submission = False
            elif not final_items:
                st.error("Please enter at least one valid item.")
                valid_submission = False
            elif abs(final_calculated_total - grand_total) > 0.02 and final_calculated_total != grand_total:
                 st.info(f"Using entered Grand Total of RM {grand_total:.2f}. Note: Item total was RM {final_calculated_total:.2f}.")
            
            if valid_submission:
                new_receipt = {
                    'merchant_name': merchant_name,
                    'date': receipt_date.strftime('%Y-%m-%d'),
                    'items': final_items,
                    'grand_total': grand_total
                }
                st.session_state.receipts_data.append(new_receipt)
                
                save_receipts_data(st.session_state.receipts_data)
                
                st.success(f"✅ Receipt from **{merchant_name}** saved successfully!")
                
                st.session_state.manual_items = [{'item_name': '', 'quantity': 1, 'unit_price': 0.0, 'total_price': 0.0}]
                st.rerun()


# ---------------------------------------------------------
# 🏁 Main App 
# ---------------------------------------------------------
def main():
    st.markdown('<div class="main-header">🛒 Purchase Performance Analytics System</div>', unsafe_allow_html=True)
    
    if st.session_state.item_database is None:
        st.session_state.item_database = load_item_database()

    page = st.sidebar.radio("Navigate", ["Upload Receipt", "Enter Manually", "Analytics Dashboard", "Settings"])

    if page == "Upload Receipt":
        st.markdown('<div class="sub-header">📤 Upload Receipt</div>', unsafe_allow_html=True)
        
        # 1. File Uploader
        uploaded_file = st.file_uploader(
            "Upload a receipt image", 
            type=["png", "jpg", "jpeg"], 
            key="receipt_uploader_key"
        )
        
        # KEY FIX: Store the uploaded file object persistently
        if uploaded_file is not None:
            # Check if a *new* file has been uploaded (using file_id for stability)
            if st.session_state.current_uploaded_file is None or st.session_state.current_uploaded_file.file_id != uploaded_file.file_id:
                 st.session_state.current_uploaded_file = uploaded_file
                 st.session_state.temp_extracted_receipt = None # Clear old review state if a new file is uploaded
        
        # --- PHASE 1: EXTRACTION (Runs if a file is present and review is NOT active) ---
        if st.session_state.current_uploaded_file and st.session_state.temp_extracted_receipt is None:
            
            file_to_process = st.session_state.current_uploaded_file
            
            image = Image.open(file_to_process)
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Original Image", use_container_width=True)
            
            # --- The Extraction Button ---
            if st.button("Extract Data", key="extract_btn_upload"):
                
                with st.spinner("Preprocessing..."):
                    processed_image = preprocess_image(image_cv)
                    # Convert processed image to bytes for persistent storage in session state
                    is_success, buffer = cv2.imencode('.png', processed_image)
                    if is_success:
                        st.session_state.processed_img_bytes = buffer.tobytes()
                
                with col2:
                    # Display processed image from stored bytes
                    if st.session_state.processed_img_bytes:
                         st.image(st.session_state.processed_img_bytes, caption="Processed Image", use_container_width=True)
                    else:
                         st.info("Processed image generated.")

                with st.spinner("Running OCR..."):
                    text = perform_ocr(processed_image)
                    st.session_state.raw_ocr_text = text 
                    
                st.text_area("📝 Raw OCR Output", text, height=200)

                with st.spinner("Extracting structured data..."):
                    receipt_data = extract_receipt_data(text, st.session_state.item_database)
                    
                if receipt_data['items']:
                    st.session_state.temp_extracted_receipt = receipt_data
                    st.success("✅ Data extracted. Please review and confirm below.")
                    st.rerun() # RERUN to switch to the review phase
                else:
                    st.warning("No items detected. Please check the raw OCR output or try manual entry.")
        
        # --- PHASE 2: REVIEW (Only runs if extracted data is in state) ---
        if st.session_state.temp_extracted_receipt:
            temp_data = st.session_state.temp_extracted_receipt
            
            st.markdown("---")
            st.subheader("🧐 Review and Edit Extracted Data")

            # --- Display stored visuals for reference ---
            if 'processed_img_bytes' in st.session_state and st.session_state.processed_img_bytes:
                col_display = st.columns(2)
                
                with col_display[0]:
                    try:
                        # Use the persistently stored file object
                        original_image = Image.open(st.session_state.current_uploaded_file)
                        st.image(original_image, caption="Original Image", use_container_width=True)
                    except:
                        st.info("Original image preview not available.")
                        
                with col_display[1]:
                    st.image(st.session_state.processed_img_bytes, caption="Processed Image", use_container_width=True)
                    
                if st.session_state.raw_ocr_text:
                     st.text_area("📝 Raw OCR Output (For Reference)", st.session_state.raw_ocr_text, height=150)
            st.markdown("---")
            
            with st.form("review_receipt_form"):
                
                # 1. Merchant, Date, Total Review
                col_sum = st.columns(3)
                with col_sum[0]:
                    merchant_name = st.text_input("Merchant Name", value=temp_data['merchant_name'] or "N/A")
                with col_sum[1]:
                    try:
                        date_value = datetime.strptime(temp_data['date'], '%Y-%m-%d').date()
                    except (ValueError, TypeError):
                        date_value = datetime.today().date()
                        
                    receipt_date = st.date_input("Date", value=date_value)

                with col_sum[2]:
                    grand_total = st.number_input(
                        "Grand Total (RM)", 
                        min_value=0.00, 
                        format="%.2f", 
                        value=temp_data['grand_total']
                    )

                # 2. Items Review (Editable Dataframe)
                st.markdown("##### Item Details (Editable)")
                df_items = pd.DataFrame(temp_data['items'])
                df_items_display = df_items.drop(columns=['category', 'match_score'], errors='ignore')
                
                edited_df = st.data_editor(
                    df_items_display,
                    column_config={
                        "item_name": st.column_config.TextColumn("Item Name", required=True),
                        "quantity": st.column_config.NumberColumn("Quantity", min_value=1, format="%d"),
                        "unit_price": st.column_config.NumberColumn("Unit Price (RM)", min_value=0.01, format="%.2f"),
                        "total_price": st.column_config.NumberColumn("Total Price (RM)", format="%.2f"),
                    },
                    num_rows="dynamic",
                    use_container_width=True,
                    key="extracted_item_editor" 
                )
                
                # 3. Confirmation Buttons
                col_buttons = st.columns(2)
                with col_buttons[0]:
                    save_submitted = st.form_submit_button("✅ Confirm and Save Data")
                with col_buttons[1]:
                    cancel_submitted = st.form_submit_button("❌ Cancel and Clear Review")

                if save_submitted:
                    final_items = []
                    for index, row in edited_df.iterrows():
                        try:
                            item_name = str(row['item_name']).strip()
                            qty = int(row['quantity'])
                            unit_price = float(row['unit_price'])
                            total_price = float(row['total_price'])
                        except (ValueError, TypeError):
                            continue 

                        if item_name and qty > 0 and total_price >= 0:
                            # Re-match the (potentially corrected) item name with the database
                            matched = match_item_with_database(item_name, st.session_state.item_database)
                            final_items.append({
                                'item_name': item_name,
                                'category': matched.get('category', 'Uncategorized'),
                                'quantity': qty,
                                'unit_price': unit_price,
                                'total_price': total_price,
                                'match_score': 1.0 # Set to 1.0 as it was manually confirmed
                            })
                            
                    if not final_items:
                        st.error("Please ensure at least one valid item remains in the list.")
                    else:
                        new_receipt = {
                            'merchant_name': merchant_name,
                            'date': receipt_date.strftime('%Y-%m-%d'),
                            'items': final_items,
                            'grand_total': grand_total
                        }
                        st.session_state.receipts_data.append(new_receipt)
                        save_receipts_data(st.session_state.receipts_data)
                        
                        # Clear all temp data, including the persistent file object
                        st.session_state.temp_extracted_receipt = None 
                        st.session_state.processed_img_bytes = None
                        st.session_state.raw_ocr_text = ""
                        st.session_state.current_uploaded_file = None # KEY CLEAR
                        st.success(f"✅ Receipt from **{merchant_name}** successfully saved after review!")
                        st.rerun()

                if cancel_submitted:
                    # Clear all temp data, including the persistent file object
                    st.session_state.temp_extracted_receipt = None 
                    st.session_state.processed_img_bytes = None
                    st.session_state.raw_ocr_text = ""
                    st.session_state.current_uploaded_file = None # KEY CLEAR
                    st.info("Review cancelled. Data was not saved.")
                    st.rerun()

    elif page == "Enter Manually":
        manual_input_page()

    # --- Analytics Dashboard Page ---
    elif page == "Analytics Dashboard":
        st.markdown('<div class="sub-header">📈 Analytics Dashboard</div>', unsafe_allow_html=True)
        if not st.session_state.receipts_data:
            st.info("No receipts yet. Please upload or enter data first.")
        else:
            total_spending = sum(float(r.get('grand_total', 0) or 0) for r in st.session_state.receipts_data)
            avg_spending = total_spending / len(st.session_state.receipts_data) if len(st.session_state.receipts_data) > 0 else 0
            
            col_metrics = st.columns(3)
            with col_metrics[0]:
                st.metric(label="Total Receipts", value=len(st.session_state.receipts_data))
            with col_metrics[1]:
                st.metric(label="Total Spending", value=f"RM {total_spending:.2f}")
            with col_metrics[2]:
                st.metric(label="Average Receipt", value=f"RM {avg_spending:.2f}")
            
            st.markdown("---")

            col_charts = st.columns(2)
            with col_charts[0]:
                st.plotly_chart(create_category_distribution(st.session_state.receipts_data), use_container_width=True)
            with col_charts[1]:
                st.plotly_chart(create_receipts_by_merchant_chart(st.session_state.receipts_data), use_container_width=True) 
            
            st.plotly_chart(create_spending_timeline(st.session_state.receipts_data), use_container_width=True)
            st.plotly_chart(create_top_items_chart(st.session_state.receipts_data), use_container_width=True)
            # --- INTERACTIVE AI ASSISTANT PANEL (Chat Below Input) ---
            st.markdown("### 💬 AI Assistant Panel")

            # Initialize chat memory
            if "ai_chat_history" not in st.session_state:
                st.session_state.ai_chat_history = []

            # --- Input box first (top) ---
            user_input = st.text_input(
                "💭 Ask a question or request advice:",
                key="ai_user_query",
                placeholder="e.g., How can I reduce my spending next month?",
            )

            col_send, col_ask = st.columns([1, 3])
            with col_send:
                ask_clicked = st.button("Send", key="ai_user_send")
            with col_ask:
                if st.button("💡 Ask AI for Personalized Insights", key="ai_advice_btn"):
                    ai_message = generate_ai_predictions(st.session_state.receipts_data)
                    st.session_state.ai_chat_history.append(("🤖 AI Assistant", ai_message))
                    st.rerun()

            # --- Process user input ---
            if ask_clicked and user_input.strip():
                question = user_input.lower()
                response = ""

                # 🪙 1. Saving & Budget Advice
                if any(k in question for k in ["save", "reduce", "budget", "cut", "limit", "spend less"]):
                    response = (
                        "💰 Based on your spending data, consider setting a weekly spending cap "
                        "and planning grocery lists ahead of time. Bulk buying and switching to store brands "
                        "could save up to **10–15%** of your monthly total."
                    )

                # 📈 2. Spending Trend Analysis
                elif any(k in question for k in ["trend", "spending", "increase", "decrease", "pattern", "change"]):
                    response = (
                        "📊 Your spending trend appears stable overall. However, essential categories like groceries "
                        "and household items show slight month-over-month growth. Consider tracking big receipts weekly."
                    )

                # 🛒 3. Top Spending Category
                elif any(k in question for k in ["category", "top", "most", "frequent", "spend on"]):
                    top_category = "Uncategorized"
                    if st.session_state.receipts_data:
                        category_spending = {}
                        for r in st.session_state.receipts_data:
                            for i in r['items']:
                                category_spending[i['category']] = category_spending.get(i['category'], 0) + i['total_price']
                        if category_spending:
                            top_category = max(category_spending, key=category_spending.get)
                    response = (
                        f"🛍️ You spend the most on **{top_category}**. Setting a limit here could give the biggest savings. "
                        "Try tracking this category more closely next month."
                    )

                # 🧾 4. Merchant Frequency
                elif any(k in question for k in ["merchant", "shop", "store", "buy from", "where"]):
                    merchant_counts = {}
                    for r in st.session_state.receipts_data:
                        merchant = r['merchant_name'] if r['merchant_name'] else "Unknown"
                        merchant_counts[merchant] = merchant_counts.get(merchant, 0) + 1
                    if merchant_counts:
                        top_merchant = max(merchant_counts, key=merchant_counts.get)
                        response = f"🏪 You shop most frequently at **{top_merchant}**. You could compare prices or loyalty rewards there."
                    else:
                        response = "🏪 I couldn’t detect any merchants yet. Upload more receipts to see detailed trends."

                # 📅 5. Day or Time-Based Spending
                elif any(k in question for k in ["day", "week", "time", "when", "sunday", "monday"]):
                    try:
                        df = pd.DataFrame(st.session_state.receipts_data)
                        df["date"] = pd.to_datetime(df["date"])
                        df["weekday"] = df["date"].dt.day_name()
                        top_day = df["weekday"].mode()[0]
                        response = f"🗓️ You tend to spend most on **{top_day}**. Planning ahead might help avoid impulse weekend buys."
                    except:
                        response = "🗓️ I need a few more dated receipts to analyze your spending days accurately."

                # 🔮 6. Forecasting / Next Month Prediction
                elif any(k in question for k in ["forecast", "next month", "future", "predict"]):
                    data = st.session_state.receipts_data
                    total_spending = sum(r['grand_total'] for r in data)
                    if len(data) > 1:
                        df_dates = pd.to_datetime(pd.DataFrame(data)['date'])
                        span_days = (df_dates.max() - df_dates.min()).days or 1
                        daily_avg = total_spending / span_days
                        monthly_forecast = daily_avg * 30
                        response = (
                            f"📆 Based on your data, you’re likely to spend around **RM {monthly_forecast:.2f}** next month. "
                            "Try setting a budget 10% lower to encourage saving."
                        )
                    else:
                        response = "📆 I’ll need at least two receipts from different days to forecast accurately."

                # 🧩 Default General Advice
                else:
                    response = (
                        "🤖 Keep monitoring your receipts! Try comparing your top categories each week "
                        "to find saving opportunities. Type 'forecast' or 'top category' to get more insights."
                    )

                # Store and rerun
                st.session_state.ai_chat_history.append(("🧑 You", user_input))
                st.session_state.ai_chat_history.append(("🤖 AI Assistant", response))
                st.rerun()


                # --- Simple offline AI logic ---
                if "save" in question or "reduce" in question or "budget" in question:
                    response = "💰 Based on your data, try focusing on your top category. Plan purchases weekly, compare brands, and buy in bulk to reduce costs."
                elif "trend" in question or "spending" in question:
                    response = "📊 Your spending pattern seems stable, but groceries and essentials might be trending upward."
                elif "category" in question or "top" in question:
                    top_category = "Uncategorized"
                    if st.session_state.receipts_data:
                        category_spending = {}
                        for r in st.session_state.receipts_data:
                            for i in r['items']:
                                category_spending[i['category']] = category_spending.get(i['category'], 0) + i['total_price']
                        if category_spending:
                            top_category = max(category_spending, key=category_spending.get)
                    response = f"🛍️ Your top spending category is **{top_category}**. Consider comparing unit prices or setting limits in this category."
                else:
                    response = "🤖 Keep monitoring your monthly spending patterns. Aim to spend 5–10% less in your top category next month."

                st.session_state.ai_chat_history.append(("🧑 You", user_input))
                st.session_state.ai_chat_history.append(("🤖 AI Assistant", response))
                st.rerun()

            # --- Display chat history (latest at the bottom) ---
            st.markdown("---")
            st.markdown("#### 💬 Chat History")

            chat_container = st.container()

            with chat_container:
                for sender, message in st.session_state.ai_chat_history:
                    bg_color = "#E6F2FF" if sender == "🤖 AI Assistant" else "#F2F2F2"
                    border_color = "#A0C8FF" if sender == "🤖 AI Assistant" else "#CCCCCC"
                    st.markdown(
                        f"""
                        <div style="
                            background-color:{bg_color};
                            color:#000000;
                            padding:12px 15px;
                            border-radius:10px;
                            margin-bottom:10px;
                            border:1px solid {border_color};
                        ">
                        <b>{sender}</b><br>{message}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Auto-scroll to bottom
            st.markdown(
                """
                <script>
                var chatContainer = window.parent.document.querySelector('.stApp');
                if (chatContainer) chatContainer.scrollTop = chatContainer.scrollHeight;
                </script>
                """,
                unsafe_allow_html=True,
            )


    # --- Settings Page ---
    elif page == "Settings":
        st.markdown('<div class="sub-header">⚙️ Settings</div>', unsafe_allow_html=True)
        
        st.subheader("Data Management")

        # --- MOVED UP: 📥 Database Operations ---
        st.markdown("##### 📥 Database Operations")
        
        # Display Stored Receipts Metric
        st.metric("Stored Receipts", len(st.session_state.receipts_data))

        # Download Button
        if st.session_state.receipts_data:
            df_full_save = pd.DataFrame(st.session_state.receipts_data)
            df_full_save['items'] = df_full_save['items'].apply(json.dumps)
            
            # Encode CSV data for download
            csv_data = df_full_save.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download All Receipts Data (CSV)",
                data=csv_data,
                file_name=DATA_FILE,
                mime='text/csv',
            )
        
        # Clear All Data Button
        if st.button("🗑️ Clear All Data (Caution! This clears everything)"):
            st.session_state.receipts_data = []
            st.session_state.manual_items = [{'item_name': '', 'quantity': 1, 'unit_price': 0.0, 'total_price': 0.0}]
            st.session_state.temp_extracted_receipt = None 
            st.session_state.current_uploaded_file = None
            save_receipts_data(st.session_state.receipts_data)
            st.success("All data cleared!")
            st.rerun()

        st.markdown("---") # Separator between database ops and single delete

        # --- MOVED DOWN: 🗑️ Delete Specific Receipt ---
        st.markdown("##### 🗑️ Delete Specific Receipt")
        if st.session_state.receipts_data:
            df_full = pd.DataFrame(st.session_state.receipts_data)
            df_full['items_summary'] = df_full['items'].apply(lambda x: f"{len(x)} items")
            df_display = df_full[['merchant_name', 'date', 'grand_total', 'items_summary']].copy()
            df_display.columns = ['Merchant Name', 'Date', 'Total (RM)', 'Items Count']
            
            st.markdown("##### Current Receipts for Reference (Index is on the far left)")
            st.dataframe(df_display, use_container_width=True) 
            st.markdown("---")
            
            selection_list = [
                f"{i}: {row['Merchant Name']} (RM{row['Total (RM)']:.2f}, {row['Date']})"
                for i, row in df_display.iterrows()
            ]
            
            st.caption("Select the index number and details of the receipt you want to delete.")
            
            selected_option = st.selectbox(
                "Select Receipt to Delete:", 
                options=["--- Select a Receipt ---"] + selection_list,
                key="receipt_selector"
            )

            if selected_option != "--- Select a Receipt ---":
                try:
                    selected_index = int(selected_option.split(':')[0])
                except ValueError:
                    st.error("Error determining index.")
                    selected_index = -1
                    
                if selected_index >= 0:
                    selected_receipt_data = df_display.loc[selected_index]
                    
                    st.warning(f"You are about to delete receipt **{selected_index}**: {selected_receipt_data['Merchant Name']} (RM{selected_receipt_data['Total (RM)']:.2f}).")

                    if st.button("🚨 CONFIRM PERMANENT DELETE", key="confirm_delete_btn"):
                        temp_df = pd.DataFrame(st.session_state.receipts_data)
                        temp_df = temp_df.drop(selected_index).reset_index(drop=True)
                        st.session_state.receipts_data = temp_df.to_dict('records')
                        
                        save_receipts_data(st.session_state.receipts_data)
                        st.success(f"✅ Receipt from {selected_receipt_data['Merchant Name']} deleted successfully!")
                        st.rerun()
            

        else:
            st.info("No receipts found in the database to manage.")
        
        st.markdown("---")
        st.subheader("Item Database")
        if st.session_state.item_database is not None:
            st.dataframe(st.session_state.item_database, use_container_width=True)
        else:
            st.error("Item database not loaded.")

if __name__ == "__main__":
    main()