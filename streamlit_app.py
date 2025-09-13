import streamlit as st
import pandas as pd
import re, string
import nltk
import contractions
from joblib import load
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import chardet
import io

# ---- One-time NLTK setup (downloads if missing; cached between runs) ----
@st.cache_resource
def _ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab")
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")
    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        nltk.download("omw-1.4")
    return True

_ = _ensure_nltk()

# ---- Load artifacts (model + vectorizer) ----
@st.cache_resource
def load_artifacts():
    mdl = load("SVM.joblib")
    vect = load("tfidf_vectorizer.joblib")
    return mdl, vect

model, vectorizer = load_artifacts()

# ---- NLP tools ----
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
APPLY_SPELL_CORRECTION = False  # toggle if you later add TextBlob

# ---- Clean function ----
def clean_text(text: str) -> str:
    text = contractions.fix(str(text))                                 # expand "don't" -> "do not"
    text = text.encode("ascii", errors="ignore").decode()              # drop non-ascii/garbled
    text = re.sub("[%s]" % re.escape(string.punctuation), " ", text)   # remove punctuation
    text = text.lower()                                                # lowercase
    text = re.sub(r"\d+", "", text)                                    # remove numbers
    words = word_tokenize(text)                                        # tokenize
    # optional spell correction (off by default)
    if APPLY_SPELL_CORRECTION:
        from textblob import TextBlob
        words = [str(TextBlob(w).correct()) for w in words]
    cleaned = [lemmatizer.lemmatize(w) for w in words
               if w not in stop_words and len(w) > 1]
    return " ".join(cleaned)

# ---- CSV Reading function with encoding detection ----
def read_csv_with_encoding(file):
    """Read CSV file with automatic encoding detection"""
    try:
        # First, try to detect encoding
        raw_data = file.read()
        file.seek(0)  # Reset file pointer
        
        # Detect encoding
        detected = chardet.detect(raw_data)
        encoding = detected['encoding'] if detected['confidence'] > 0.7 else 'utf-8'
        
        # Try detected encoding first
        try:
            file.seek(0)
            df = pd.read_csv(io.StringIO(raw_data.decode(encoding)))
            return df, encoding
        except (UnicodeDecodeError, LookupError):
            pass
        
        # Try common encodings
        encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
        
        for encoding in encodings_to_try:
            try:
                file.seek(0)
                df = pd.read_csv(io.StringIO(raw_data.decode(encoding)))
                return df, encoding
            except (UnicodeDecodeError, LookupError):
                continue
        
        # If all encodings fail, try with error handling
        try:
            file.seek(0)
            df = pd.read_csv(io.StringIO(raw_data.decode('utf-8', errors='replace')))
            return df, 'utf-8 (with error replacement)'
        except Exception:
            raise ValueError("Could not decode the file with any common encoding")
            
    except Exception as e:
        raise ValueError(f"Error reading file: {str(e)}")

@st.cache_data
def predict_batch(texts):
    """Predict sentiment for a list of texts"""
    results = []
    for text in texts:
        if pd.isna(text) or str(text).strip() == '':
            results.append('unknown')
        else:
            cleaned = clean_text(str(text))
            if cleaned.strip() == '':
                results.append('unknown')
            else:
                features = vectorizer.transform([cleaned])
                pred = model.predict(features)[0]
                results.append(pred.lower())
    return results

def find_review_column(df):
    """Find review column by looking for variations of 'review'"""
    possible_names = ['review', 'Review', 'REVIEW']
    for col in df.columns:
        if col in possible_names:
            return col
    return None

# ---- UI Configuration ----
st.set_page_config(page_title="Sentiment Analysis", page_icon="📊", layout="wide")

# ---- Initialize session state ----
if "total_reviews" not in st.session_state:
    st.session_state.total_reviews = 0
    st.session_state.positive = 0
    st.session_state.negative = 0
if "csv_results" not in st.session_state:
    st.session_state.csv_results = None
# Initialize session state for preserving inputs and results
if "user_text_input" not in st.session_state:
    st.session_state.user_text_input = ""
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "last_cleaned_text" not in st.session_state:
    st.session_state.last_cleaned_text = ""
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None
if "csv_analysis_done" not in st.session_state:
    st.session_state.csv_analysis_done = False

# ---- Sidebar Navigation ----
st.sidebar.title("📊 Navigation")
page = st.sidebar.selectbox("Choose a page:", ["✍️ Review Prediction", "📁 CSV Analysis"])

# ---- Review Prediction Page ----
if page == "✍️ Review Prediction":
    st.title("🍟 McDonald's Review Sentiment Classifier")
    st.write("Here's a demo of how the model classifies reviews into **Positive** or **Negative**.")

    # Demo Sentiment Table
    st.subheader("🧾 Demo Sentiment Table")
    
    # Create demo data
    demo_reviews = [
        'The fries were cold and soggy, very disappointed.',
        'Fast delivery, the burger was still hot when it arrived!',
        'Customer service was rude and unhelpful.',
        'The app is easy to use, and ordering was smooth.',
        'The drink spilled inside the bag, really bad experience.',
        'Great value for the price, I will definitely order again.',
        'Long waiting time and the order was still wrong',
        'The new menu item is delicious, highly recommended!',
        'Packaging was terrible, food looked messy when it arrived.',
        'Friendly staff, clean restaurant, and tasty food.'
        
    ]
    
    actual_sentiments = [
        'Negative', 'Positive', 'Negative', 'Positive', 'Negative', 
        'Positive', 'Negative', 'Positive', 'Negative', 'Positive'
    ]
    
    # Generate dynamic predictions using the loaded model
    predicted_sentiments = []
    for review in demo_reviews:
        cleaned = clean_text(review)
        if cleaned.strip() == '':
            predicted_sentiments.append('Unknown')
        else:
            features = vectorizer.transform([cleaned])
            pred = model.predict(features)[0]
            predicted_sentiments.append(pred.capitalize())  # Capitalize to match format
    
    demo_data = {
        'Review': demo_reviews,
        'Actual Sentiment': actual_sentiments,
        'Predicted Sentiment': predicted_sentiments
    }
    
    demo_df = pd.DataFrame(demo_data)
    
    # Display the table with styling
    st.dataframe(
        demo_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Review": st.column_config.TextColumn("Review", width="large"),
            "Actual Sentiment": st.column_config.TextColumn("Actual Sentiment", width="medium"),
            "Predicted Sentiment": st.column_config.TextColumn("Predicted Sentiment (BERT)", width="medium")
        }
    )
    
    st.markdown("---")

    # Text input
    st.write("Now try typing your own review below to see the prediction in action!")
    user_text = st.text_area("✍️ Type a review here:", value=st.session_state.user_text_input, height=160, placeholder="e.g., The fries were crispy and the staff were super friendly!", key="review_text_area")
    
    # Update session state when text changes
    st.session_state.user_text_input = user_text

    col1, col2 = st.columns(2)
    with col1:
        predict_btn = st.button("🔮 Predict")
    
    # Show last prediction if it exists and no new prediction is being made
    if predict_btn:
        if not user_text.strip():
            st.warning("Please enter a review first.")
        else:
            cleaned = clean_text(user_text)
            features = vectorizer.transform([cleaned])
            pred = model.predict(features)[0]  # e.g., "positive" or "negative"

            # Store prediction in session state
            st.session_state.last_prediction = pred
            st.session_state.last_cleaned_text = cleaned

            if pred.lower() == "positive":
                st.success("✅ Predicted Sentiment: **Positive** 😃")
                st.session_state.positive += 1
            else:
                st.error("❌ Predicted Sentiment: **Negative** 😞")
                st.session_state.negative += 1
            
            st.session_state.total_reviews += 1

            # Show the cleaned text
            with st.expander("See cleaned text used for prediction"):
                st.code(cleaned)
    
    # Display last prediction if it exists (when returning to page)
    elif st.session_state.last_prediction is not None and st.session_state.user_text_input.strip():
        if st.session_state.last_prediction.lower() == "positive":
            st.success("✅ Predicted Sentiment: **Positive** 😃")
        else:
            st.error("❌ Predicted Sentiment: **Negative** 😞")
        
        # Show the cleaned text (useful for demo)
        with st.expander("See cleaned text used for prediction"):
            st.code(st.session_state.last_cleaned_text)

# ---- CSV Analysis Page ----
elif page == "📁 CSV Analysis":
    st.title("📁 CSV Analysis")
    st.write("Upload a CSV file with reviews to analyze sentiment.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    # Check if this is a new file or returning to previous results
    if uploaded_file is not None:
        current_file_name = uploaded_file.name
        
        # If it's a new file, process it
        if (st.session_state.uploaded_file_name != current_file_name or 
            st.session_state.csv_results is None):
            
            try:
                # Read CSV with encoding detection
                with st.spinner("Reading file and detecting encoding..."):
                    df, encoding_used = read_csv_with_encoding(uploaded_file)
                
                st.success(f"✅ File uploaded successfully! Found {len(df)} rows.")
                st.info(f"📝 Encoding detected: {encoding_used}")
                
                st.subheader("📋 Data Preview")
                st.dataframe(df.head(10))
                
                # Check for review column
                review_column = find_review_column(df)
                
                if review_column is None:
                    st.subheader("📊 Analysis Results")
                    st.error("❌ No review column found! Please make sure your CSV has a column named 'review', 'Review', or 'REVIEW'.")
                    # Clear previous results
                    st.session_state.csv_results = None
                    st.session_state.csv_analysis_done = False
                else:
                    with st.spinner("Analyzing sentiments... This may take a moment for large files."):
                        # Get texts from review column
                        texts = df[review_column].tolist()
                        
                        # Predict sentiments
                        sentiments = predict_batch(texts)
                        
                        # Create results dataframe
                        results_df = df.copy()
                        results_df['sentiment'] = sentiments
                        
                        # Add cleaned text column
                        cleaned_texts = []
                        for text in texts:
                            if pd.isna(text) or str(text).strip() == '':
                                cleaned_texts.append('')
                            else:
                                cleaned_texts.append(clean_text(str(text)))
                        
                        results_df['cleaned_review'] = cleaned_texts
                        
                        # Store in session state
                        st.session_state.csv_results = results_df
                        st.session_state.uploaded_file_name = current_file_name
                        st.session_state.csv_analysis_done = True
            
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.markdown("""
                **Possible solutions:**
                - Make sure your file is a valid CSV
                - Try saving your file with UTF-8 encoding
                - Check if the file is corrupted
                - Try a different CSV file format
                
                **Common issues:**
                - File contains special characters or non-standard encoding
                - File is not properly formatted as CSV
                - File is too large or corrupted
                """)
                
                # Provide encoding troubleshooting
                if "codec can't decode" in str(e) or "encoding" in str(e).lower():
                    st.markdown("""
                    **Encoding Issue Detected:**
                    Your file seems to have encoding problems. Try these steps:
                    1. Open your CSV in Excel or Google Sheets
                    2. Save it as "CSV UTF-8" format
                    3. Upload the newly saved file
                    """)
                    
                    # Show raw bytes for debugging
                    if st.checkbox("🔧 Show file debugging info (Advanced)"):
                        try:
                            uploaded_file.seek(0)
                            raw_bytes = uploaded_file.read(1000)
                            st.code(f"First 1000 bytes: {raw_bytes}")
                            
                            # Try to detect encoding
                            detected = chardet.detect(raw_bytes)
                            st.write(f"Detected encoding: {detected}")
                        except:
                            st.write("Could not analyze file bytes")
                # Clear results on error
                st.session_state.csv_results = None
                st.session_state.csv_analysis_done = False
    
    # Display results if they exist (either from new analysis or previous session)
    if st.session_state.csv_results is not None and st.session_state.csv_analysis_done:
        results_df = st.session_state.csv_results
        
        # Show results using Dashboard style
        st.subheader("📊 Analysis Results")
        
        # Get sentiment counts (excluding unknown/empty for main analysis)
        sentiment_counts = results_df['sentiment'].value_counts()
        csv_positive = sentiment_counts.get('positive', 0)
        csv_negative = sentiment_counts.get('negative', 0)
        csv_unknown = sentiment_counts.get('unknown', 0)
        
        # Display current statistics (Dashboard style)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Reviews", len(results_df))
        
        with col2:
            st.metric("Positive Reviews", csv_positive)
        
        with col3:
            st.metric("Negative Reviews", csv_negative)
        
        with col4:
            st.metric("Unknown/Empty", csv_unknown)
        
        # Create visualizations (Dashboard style)
        total_analyzed = csv_positive + csv_negative
        
        if total_analyzed > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                # Pie Chart
                labels = ['Positive', 'Negative']
                values = [csv_positive, csv_negative]
                colors = ['#2E8B57', '#DC143C']
                
                fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4)])
                fig_pie.update_traces(marker=dict(colors=colors, line=dict(color='white', width=2)))
                fig_pie.update_layout(title="Sentiment Distribution", font=dict(size=14), showlegend=True, height=400
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Bar Chart
                fig_bar = go.Figure(data=[go.Bar(name='Sentiment', x=labels, y=values, marker_color=colors, text=values, textposition='auto')])
                fig_bar.update_layout(title="Sentiment Count", xaxis_title="Sentiment", yaxis_title="Number of Reviews", height=400, showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Percentage breakdown
            pos_percent = (csv_positive / total_analyzed) * 100
            neg_percent = (csv_negative / total_analyzed) * 100
            
            st.markdown("### 📈 Analysis Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Positive Reviews:** {pos_percent:.1f}% ({csv_positive} reviews)")
            with col2:
                st.info(f"**Negative Reviews:** {neg_percent:.1f}% ({csv_negative} reviews)")
            
            # Show unknown/empty info if any
            if csv_unknown > 0:
                unknown_percent = (csv_unknown / len(results_df)) * 100
                st.warning(f"**Note:** {csv_unknown} reviews ({unknown_percent:.1f}%) were empty or could not be analyzed.")
        
        else:
            st.info("📝 No valid reviews found for sentiment analysis.")
        
        # Show detailed results
        st.subheader("📋 Detailed Results")
        
        # Filter options
        sentiment_filter = st.selectbox("Filter by sentiment:", ['All', 'Positive', 'Negative', 'Unknown'])
        
        if sentiment_filter != 'All':
            filtered_df = results_df[results_df['sentiment'] == sentiment_filter.lower()]
        else:
            filtered_df = results_df
        
        st.dataframe(filtered_df, use_container_width=True)
        
        # Download results
        csv = results_df.to_csv(index=False)
        st.download_button(label="📥 Download Results as CSV", data=csv, file_name="sentiment_analysis_results.csv", mime="text/csv"
        )