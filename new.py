import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import speech_recognition as sr
import os
from google import genai


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Autonomous Data Analysis Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Autonomous Data Analysis Agent")
st.write("Upload your dataset and ask questions using Text or Voice!")


# =========================================================
# GEMINI API KEY
# =========================================================

API_KEY = os.getenv("GEMINI_API_KEY")


if not API_KEY:
    st.error(
        "❌ Gemini API key not found. "
        "Please set GEMINI_API_KEY in your environment variables."
    )
    st.stop()


# =========================================================
# GEMINI CLIENT
# =========================================================

client = genai.Client(api_key=API_KEY)


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📂 Upload CSV or Excel File",
    type=["csv", "xlsx"]
)


# =========================================================
# LOAD FILE
# =========================================================

def load_file(file):

    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)

    elif file.name.lower().endswith(".xlsx"):
        return pd.read_excel(file)

    return None


# =========================================================
# VOICE INPUT
# =========================================================

def get_voice_input():

    recognizer = sr.Recognizer()

    try:

        with sr.Microphone() as source:

            st.info("🎤 Listening... Speak now")

            recognizer.adjust_for_ambient_noise(
                source,
                duration=1
            )

            audio = recognizer.listen(
                source,
                timeout=10,
                phrase_time_limit=20
            )

        return recognizer.recognize_google(audio)

    except sr.WaitTimeoutError:

        return "No speech detected."

    except sr.UnknownValueError:

        return "Could not understand the audio."

    except sr.RequestError:

        return "Speech recognition service unavailable."

    except Exception as e:

        return f"Voice error: {e}"


# =========================================================
# DATASET ANALYSIS
# =========================================================

def get_dataset_context(df):

    columns = list(df.columns)

    rows = df.shape[0]

    cols = df.shape[1]

    missing = int(df.isnull().sum().sum())

    # Full dataset
    full_data = df.to_string(index=False)

    # Statistics
    numeric_cols = df.select_dtypes(
        include="number"
    ).columns

    if len(numeric_cols) > 0:

        statistics = df[numeric_cols].describe().to_string()

    else:

        statistics = "No numeric columns."

    # Categorical value counts
    categorical_info = ""

    categorical_cols = df.select_dtypes(
        exclude="number"
    ).columns

    for column in categorical_cols:

        categorical_info += (
            f"\n{column} value counts:\n"
        )

        categorical_info += (
            df[column]
            .value_counts(dropna=False)
            .to_string()
        )

        categorical_info += "\n\n"

    return (
        columns,
        rows,
        cols,
        missing,
        full_data,
        statistics,
        categorical_info
    )


# =========================================================
# ASK GEMINI
# =========================================================

def ask_gemini(df, question):

    (
        columns,
        rows,
        cols,
        missing,
        full_data,
        statistics,
        categorical_info
    ) = get_dataset_context(df)

    prompt = f"""
You are an expert data analyst.

Analyze the COMPLETE dataset provided below.

DATASET INFORMATION
===================

Total Rows: {rows}

Total Columns: {cols}

Column Names:
{columns}

Missing Values:
{missing}


COMPLETE DATASET
================

{full_data}


NUMERIC STATISTICS
==================

{statistics}


CATEGORICAL VALUE COUNTS
========================

{categorical_info}


USER QUESTION
=============

{question}


IMPORTANT INSTRUCTIONS
======================

1. Use the COMPLETE dataset, not only the first few rows.

2. If the user asks for a count, calculate it from the complete dataset.

3. If the question is about a categorical column such as sex, smoker,
   region, etc., use the value counts provided above.

4. Do NOT assume that the first 10 rows represent the entire dataset.

5. Do NOT invent information.

6. Give an exact answer whenever the dataset contains enough information.

7. Explain the calculation briefly when appropriate.

8. Answer in simple and clear language.

9. If the requested information genuinely cannot be determined,
   clearly explain why.

10. Treat the dataset as the source of truth.
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )

    return response.text


# =========================================================
# MAIN APP
# =========================================================

if uploaded_file:

    df = load_file(uploaded_file)

    if df is None:

        st.error("❌ Could not load the file.")

        st.stop()


    # =====================================================
    # CLEAN COLUMN NAMES
    # =====================================================

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.replace(" ", "_")
    )


    # =====================================================
    # DATASET PREVIEW
    # =====================================================

    st.subheader("📊 Dataset Preview")

    st.dataframe(
        df,
        use_container_width=True
    )


    # =====================================================
    # DATASET INFORMATION
    # =====================================================

    st.sidebar.header("📌 Dataset Information")

    st.sidebar.write(
        "Rows:",
        df.shape[0]
    )

    st.sidebar.write(
        "Columns:",
        df.shape[1]
    )

    st.sidebar.write(
        "Missing Values:",
        int(df.isnull().sum().sum())
    )

    st.sidebar.subheader("Column Names")

    for column in df.columns:

        st.sidebar.write(
            f"• {column}"
        )


    # =====================================================
    # INPUT MODE
    # =====================================================

    st.subheader("📝 Ask Your Dataset")

    mode = st.radio(
        "Choose Input Mode",
        ["Text", "Voice"],
        horizontal=True
    )

    query = ""


    # =====================================================
    # TEXT INPUT
    # =====================================================

    if mode == "Text":

        query = st.text_input(
            "💬 Ask your question",
            placeholder="Example: How many males are in the dataset?"
        )


    # =====================================================
    # VOICE INPUT
    # =====================================================

    else:

        if st.button("🎤 Start Voice Input"):

            query = get_voice_input()

            st.success(
                f"You said: {query}"
            )


    # =====================================================
    # GEMINI ANALYSIS
    # =====================================================

    if query:

        with st.spinner(
            "🤖 Gemini is analyzing the complete dataset..."
        ):

            try:

                # -------------------------------------------------
                # EXACT DATASET CALCULATIONS
                # -------------------------------------------------

                answer = None

                question_lower = query.lower()


                # =================================================
                # MALE COUNT
                # =================================================

                if (
                    "how many male" in question_lower
                    or "number of male" in question_lower
                    or "count of male" in question_lower
                ):

                    for col in df.columns:

                        if col.lower() in [
                            "sex",
                            "gender"
                        ]:

                            male_count = (
                                df[col]
                                .astype(str)
                                .str.lower()
                                .eq("male")
                                .sum()
                            )

                            answer = (
                                f"According to the complete dataset, "
                                f"there are **{male_count} male members**."
                            )

                            break


                # =================================================
                # FEMALE COUNT
                # =================================================

                if (
                    answer is None
                    and (
                        "how many female" in question_lower
                        or "number of female" in question_lower
                        or "count of female" in question_lower
                    )
                ):

                    for col in df.columns:

                        if col.lower() in [
                            "sex",
                            "gender"
                        ]:

                            female_count = (
                                df[col]
                                .astype(str)
                                .str.lower()
                                .eq("female")
                                .sum()
                            )

                            answer = (
                                f"According to the complete dataset, "
                                f"there are **{female_count} female members**."
                            )

                            break


                # =================================================
                # GEMINI FOR OTHER QUESTIONS
                # =================================================

                if answer is None:

                    answer = ask_gemini(
                        df,
                        query
                    )


                # =================================================
                # DISPLAY ANSWER
                # =================================================

                st.subheader(
                    "🧠 Gemini AI Answer"
                )

                st.markdown(answer)


            except Exception as e:

                st.error(
                    f"❌ Gemini Error: {e}"
                )


    # =====================================================
    # VISUALIZATION
    # =====================================================

    st.divider()

    st.subheader("📈 Data Visualization")

    numeric_cols = (
        df.select_dtypes(
            include="number"
        )
        .columns
        .tolist()
    )


    if numeric_cols:

        selected_col = st.selectbox(
            "Select Numeric Column",
            numeric_cols
        )

        chart_type = st.selectbox(
            "Select Chart Type",
            [
                "Line Chart",
                "Bar Chart",
                "Histogram"
            ]
        )


        fig, ax = plt.subplots()


        if chart_type == "Line Chart":

            ax.plot(
                df[selected_col]
            )

            ax.set_title(
                f"{selected_col} Trend"
            )


        elif chart_type == "Bar Chart":

            data = df[selected_col].head(50)

            ax.bar(
                range(len(data)),
                data
            )

            ax.set_title(
                f"{selected_col} Bar Chart"
            )


        elif chart_type == "Histogram":

            ax.hist(
                df[selected_col].dropna(),
                bins=20
            )

            ax.set_title(
                f"{selected_col} Distribution"
            )


        ax.set_xlabel("Index")

        ax.set_ylabel(
            selected_col
        )

        st.pyplot(fig)


    else:

        st.info(
            "No numeric columns available for visualization."
        )


else:

    st.info(
        "👆 Upload a CSV or Excel dataset to begin."
    )