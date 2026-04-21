# ===============================================================
# RANDOM FOREST PRIVACY AGENT (BLOCK 1)
# ===============================================================

import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ===============================================================
# MAIN FUNCTION
# ===============================================================

def run_random_forest():
    """Execute complete Random Forest pipeline"""

    try:
        print("=" * 90)
        print("STEP 1: LOAD DATASET")
        print("=" * 90)

        # Load dataset from DATASET folder
        file_path = "DATASET/MIMIC_IV_Trasncript.csv"

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset not found at {file_path}")

        df = pd.read_csv(file_path, low_memory=False)
        print(f"✓ Loaded: Shape {df.shape}")
        print(f"✓ Columns: {df.columns.tolist()[:5]}...")

        # ===============================================================
        # STEP 2: REMOVE DUPLICATES
        # ===============================================================
        print("\n" + "=" * 90)
        print("STEP 2: REMOVE DUPLICATES")
        print("=" * 90)

        before_rows = len(df)
        df = df.drop_duplicates()
        after_rows = len(df)
        removed = before_rows - after_rows

        print(f"✓ Before: {before_rows}, After: {after_rows}, Removed: {removed}")

        # ===============================================================
        # STEP 3: CREATE RAW DATA (Original format with patient id)
        # ===============================================================
        print("\n" + "=" * 90)
        print("STEP 3: CREATE RAW DATA")
        print("=" * 90)

        # ✅ THIS IS THE RAW DATA FORMAT
        df['raw_text'] = (
                "patient id " + df['subject_id'].astype(str) +
                " admission " + df['hadm_id'].astype(str) +
                " age " + df['anchor_age'].astype(str) +
                " gender " + df['gender'].astype(str) +
                " diagnosis " + df['description'].astype(str) +
                " prescribed " + df['drug'].astype(str) +
                " test " + df['test_name'].astype(str)
        )

        print(f"✓ Created raw data for {len(df)} records")

        # ===============================================================
        # STEP 4: CREATE SENTENCE (for feature extraction)
        # ===============================================================
        print("\n" + "=" * 90)
        print("STEP 4: CREATE SENTENCE FOR FEATURES")
        print("=" * 90)

        df['sentence'] = (
                "age " + df['anchor_age'].astype(str) +
                " gender " + df['gender'].astype(str) +
                " diagnosis " + df['description'].astype(str) +
                " prescribed " + df['drug'].astype(str) +
                " test " + df['test_name'].astype(str)
        )

        print(f"✓ Created sentences for {len(df)} records")

        # ===============================================================
        # STEP 5: PREPROCESS TEXT
        # ===============================================================
        print("\n" + "=" * 90)
        print("STEP 5: PREPROCESS TEXT")
        print("=" * 90)

        def preprocess_text(text):
            text = str(text).lower()
            text = re.sub(r'[^a-z0-9\s:/.-]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        df['clean_sentence'] = df['sentence'].apply(preprocess_text)
        print(f"✓ Preprocessed all sentences")

        # ===============================================================
        # STEP 6: FEATURE EXTRACTION
        # ===============================================================
        print("\n" + "=" * 90)
        print("STEP 6: FEATURE EXTRACTION")
        print("=" * 90)

        medical_words = [
            'pain', 'fever', 'diabetes', 'infection', 'liver', 'heart',
            'aspirin', 'vancomycin', 'insulin', 'cancer', 'fracture'
        ]
        privacy_words = ['name', 'address', 'phone', 'ssn', 'contact', 'email']
        hospital_words = ['hospital', 'clinic', 'icu', 'ward']
        name_words = ['john', 'mary', 'smith', 'raj', 'kumar']

        df['word_count'] = df['clean_sentence'].apply(lambda x: len(x.split()))
        df['char_count'] = df['clean_sentence'].apply(len)
        df['number_count'] = df['clean_sentence'].apply(lambda x: len(re.findall(r'\d+(\.\d+)?', x)))
        df['date_presence'] = df['clean_sentence'].apply(
            lambda x: 1 if re.search(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', x) else 0)
        df['time_presence'] = df['clean_sentence'].apply(lambda x: 1 if re.search(r'\b\d{1,2}:\d{2}\b', x) else 0)
        df['name_count'] = df['clean_sentence'].apply(lambda x: sum(w in x for w in name_words))
        df['location_count'] = df['clean_sentence'].apply(lambda x: sum(w in x for w in hospital_words))
        df['medical_count'] = df['clean_sentence'].apply(lambda x: sum(w in x for w in medical_words))
        df['privacy_kw_count'] = df['clean_sentence'].apply(lambda x: sum(w in x for w in privacy_words))

        feature_cols = [
            'word_count', 'char_count', 'number_count',
            'date_presence', 'time_presence',
            'name_count', 'location_count',
            'medical_count', 'privacy_kw_count'
        ]

        print(f"✓ Created {len(feature_cols)} features")

        # ===============================================================
        # STEP 7: CREATE LABELS
        # ===============================================================
        print("\n" + "=" * 90)
        print("STEP 7: CREATE LABELS (Safe=0, Sensitive=1)")
        print("=" * 90)

        def create_label(row):
            score = 0
            if row['name_count'] > 0:
                score += 1
            if row['privacy_kw_count'] >= 1:
                score += 1
            if row['number_count'] >= 2:
                score += 1
            if row['medical_count'] == 0:
                score += 1
            return 1 if score >= 2 else 0

        df['label'] = df.apply(create_label, axis=1)
        label_counts = df['label'].value_counts().to_dict()

        print(f"✓ Safe Records: {label_counts.get(0, 0)}")
        print(f"✓ Sensitive Records: {label_counts.get(1, 0)}")

        # ===============================================================
        # STEP 8: RANDOM FOREST MODEL
        # ===============================================================
        print("\n" + "=" * 90)
        print("STEP 8: RANDOM FOREST MODEL TRAINING")
        print("=" * 90)

        X = df[feature_cols]
        y = df['label']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"✓ Train set: {X_train.shape}")
        print(f"✓ Test set: {X_test.shape}")

        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)
        df['prediction'] = rf.predict(X)
        df['result'] = df['prediction'].map({1: 'Sensitive', 0: 'Safe'})

        print(f"✓ Model trained successfully")

        # ===============================================================
        # STEP 9: PRIVACY MASKING
        # ===============================================================
        print("\n" + "=" * 90)
        print("STEP 9: PRIVACY MASKING")
        print("=" * 90)

        def get_age_group(age):
            try:
                age = int(age)
                if 0 <= age <= 12:
                    return "Childhood"
                elif 13 <= age <= 19:
                    return "Adolescence"
                elif 20 <= age <= 64:
                    return "Adulthood"
                else:
                    return "Senior"
            except:
                return "Unknown"

        def age_group_replace(match):
            age = match.group(1)
            return "age " + get_age_group(age)

        def make_safe(text):
            t = str(text)
            t = re.sub(r'age (\d+)', age_group_replace, t)
            t = re.sub(r'\b(john|mary|smith|raj|kumar)\b', '[masked_name]', t)
            t = re.sub(r'\s+', ' ', t).strip()
            return t

        df['final_text'] = df['clean_sentence'].apply(make_safe)
        print(f"✓ Privacy masking applied to all records")

        # ===============================================================
        # STEP 10: METRICS
        # ===============================================================
        print("\n" + "=" * 90)
        print("STEP 10: MODEL METRICS")
        print("=" * 90)

        acc = accuracy_score(y_test, y_pred)
        pre = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"✓ Accuracy: {acc:.4f}")
        print(f"✓ Precision: {pre:.4f}")
        print(f"✓ Recall: {rec:.4f}")
        print(f"✓ F1-Score: {f1:.4f}")

        # ===============================================================
        # STEP 11: SAVE FILES
        # ===============================================================
        print("\n" + "=" * 90)
        print("STEP 11: SAVE FILES")
        print("=" * 90)

        save_folder = "RF"
        os.makedirs(save_folder, exist_ok=True)

        # ✅ FILE 1: RAW_DATA - Original format with patient id
        df[['raw_text']].to_csv(
            f"{save_folder}/RAW_DATA.csv", index=False
        )
        print(f"✓ Saved: RAW_DATA.csv")

        # ✅ FILE 2: RF_PRIVACY_RESULTS - Privacy classification
        df[['subject_id', 'hadm_id', 'anchor_age', 'result']].to_csv(
            f"{save_folder}/RF_PRIVACY_RESULTS.csv", index=False
        )
        print(f"✓ Saved: RF_PRIVACY_RESULTS.csv")

        # ✅ FILE 3: SAFE_TEXT_FOR_LLMS - Masked text
        df[['final_text']].to_csv(
            f"{save_folder}/SAFE_TEXT_FOR_LLMS.csv", index=False
        )
        print(f"✓ Saved: SAFE_TEXT_FOR_LLMS.csv")

        print(f"✓ Saved to RF/ folder")

        # ===============================================================
        # PREPARE RESPONSE
        # ===============================================================

        raw_samples = df['raw_text'].head(5).tolist()
        safe_samples = df['final_text'].head(5).tolist()

        response = {
            'status': 'success',
            'raw_data': '\n\n'.join([f"{i+1}. {s}" for i, s in enumerate(raw_samples)]),
            'privacy_results': f"✓ Safe: {label_counts.get(0, 0)} | Sensitive: {label_counts.get(1, 0)}",
            'safe_text': '\n\n'.join([f"{i+1}. {s}" for i, s in enumerate(safe_samples)]),
            'metrics': {
                'Accuracy': round(acc, 4),
                'Precision': round(pre, 4),
                'Recall': round(rec, 4),
                'F1-Score': round(f1, 4),
                'Total Records': len(df),
                'Safe Records': label_counts.get(0, 0),
                'Sensitive Records': label_counts.get(1, 0)
            }
        }

        print("\n" + "=" * 90)
        print("✓ BLOCK 1 COMPLETED SUCCESSFULLY")
        print("=" * 90 + "\n")

        return response

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}\n")
        return {
            'status': 'error',
            'error': str(e)
        }