# ===============================================================
# PROJECT: HYDRAGAN – FINAL PRIVACY SAFE VERSION
# INPUT  : APPROVED_TEXT_FOR_GAN.csv
# OUTPUT :
#   1. HYDRAGAN_SYNTHETIC_TEXT_FINAL.csv
#   2. HYDRAGAN_METRICS_FINAL.csv
# ===============================================================

import pandas as pd
import numpy as np
import random
import re
import os

random.seed(42)
np.random.seed(42)


# ===============================================================
# MAIN FUNCTION
# ===============================================================

def run_hydragan():
    """Execute HydraGAN synthetic data generation"""

    try:
        print("=" * 90)
        print("STEP 1: FILE LOADED")
        print("=" * 90)

        approved_path = "HYBRID/APPROVED_TEXT_FOR_GAN.csv"

        if not os.path.exists(approved_path):
            raise FileNotFoundError(f"Block 2 output not found. Run Block 2 first!")

        df = pd.read_csv(approved_path)

        print(f"Shape: {df.shape}")
        print(df.head().to_string())

        text_col = df.columns[0]
        real_texts = df[text_col].astype(str).str.lower().tolist()
        real_set = set(real_texts)

        # ===============================================================
        # STEP 2: LEARN PATTERNS FROM REAL DATA
        # ===============================================================
        print("\n" + "=" * 90)
        print("STEP 2: LEARN PATTERNS FROM REAL DATA")
        print("=" * 90)

        ages = set()
        genders = set()
        diagnoses = set()
        drugs = set()
        tests = set()

        for txt in real_texts:
            a = re.search(r'age\s+(.*?)\s+gender', txt)
            g = re.search(r'gender\s+(.*?)\s+diagnosis', txt)
            d = re.search(r'diagnosis\s+(.*?)\s+prescribed', txt)
            p = re.search(r'prescribed\s+(.*?)\s+test', txt)
            t = re.search(r'test\s+(.*)', txt)

            if a: ages.add(a.group(1).strip())
            if g: genders.add(g.group(1).strip())
            if d: diagnoses.add(d.group(1).strip())
            if p: drugs.add(p.group(1).strip())
            if t: tests.add(t.group(1).strip())

        ages = list(ages)
        genders = list(genders)
        diagnoses = list(diagnoses)
        drugs = list(drugs)
        tests = list(tests)

        print(f"\nLearned Patterns:")
        print(f"Age Groups : {len(ages)}")
        print(f"Gender     : {len(genders)}")
        print(f"Diagnosis  : {len(diagnoses)}")
        print(f"Drugs      : {len(drugs)}")
        print(f"Tests      : {len(tests)}")

        # ===============================================================
        # STEP 3: MULTIPLE GENERATORS (Hydra Heads)
        # ===============================================================
        print("\n" + "=" * 90)
        print("STEP 3: SETUP GENERATOR TEMPLATES")
        print("=" * 90)

        templates = [
            "age {age} gender {gender} diagnosis {diag} prescribed {drug} test {test}",
            "{gender} patient age {age} diagnosed with {diag} treated with {drug} underwent {test}",
            "patient with {diag} age {age} gender {gender} medication {drug} lab {test}",
            "{age} {gender} case diagnosis {diag} given {drug} test {test}",
            "diagnosis {diag} in {age} {gender} patient prescribed {drug} recommended {test}",
            "{gender} individual age {age} with {diag} managed using {drug} test {test}",
            "medical case age {age} gender {gender} diagnosis {diag} therapy {drug} lab {test}"
        ]

        synonym_map = {
            "prescribed": ["prescribed", "given", "treated with", "administered"],
            "test": ["test", "lab", "screening", "evaluation"]
        }

        print(f"✓ Created {len(templates)} generator templates")

        # ===============================================================
        # STEP 4: GENERATOR FUNCTION
        # ===============================================================
        print("\n" + "=" * 90)
        print("STEP 4: GENERATOR FUNCTION")
        print("=" * 90)

        def generate_sentence():
            temp = random.choice(templates)
            sent = temp.format(
                age=random.choice(ages),
                gender=random.choice(genders),
                diag=random.choice(diagnoses),
                drug=random.choice(drugs),
                test=random.choice(tests)
            )

            for key, vals in synonym_map.items():
                sent = sent.replace(key, random.choice(vals), 1)

            sent = " ".join(sent.split())
            return sent.lower()

        print("✓ Generator ready")

        # ===============================================================
        # STEP 5: DISCRIMINATOR FUNCTION
        # ===============================================================
        print("\n" + "=" * 90)
        print("STEP 5: DISCRIMINATOR FUNCTION")
        print("=" * 90)

        medical_words = {
            "diagnosis", "liver", "culture", "infection", "pain", "fever", "drug",
            "test", "respiratory", "urine", "gram", "blood", "cardiac", "coronary",
            "furosemide", "heparin", "albuterol", "albumin", "vancomycin",
            "hepatic", "coma", "acute", "stroke", "septicemia", "aspirin",
            "insulin", "arrhythmia", "pneumonia", "screening", "lab"
        }

        def discriminator(text):
            words = text.split()
            score = 0

            if len(words) >= 8:
                score += 1

            med = sum(w in medical_words for w in words)
            if med >= 2:
                score += 1

            if "age" in words and "gender" in words:
                score += 1

            return score >= 2

        print("✓ Discriminator ready")

        # ===============================================================
        # STEP 6: GENERATE PRIVACY SAFE SYNTHETIC TEXT
        # ===============================================================
        print("\n" + "=" * 90)
        print("STEP 6: GENERATE SYNTHETIC DATA")
        print("=" * 90)

        synthetic_texts = []
        synthetic_set = set()

        needed = len(real_texts)
        attempts = 0
        max_attempts = needed * 5

        print(f"Target: {needed} synthetic samples")

        while len(synthetic_texts) < needed and attempts < max_attempts:
            s = generate_sentence()
            attempts += 1

            if discriminator(s) and s not in real_set and s not in synthetic_set:
                synthetic_texts.append(s)
                synthetic_set.add(s)

                if len(synthetic_texts) % 2000 == 0:
                    print(f"Generated: {len(synthetic_texts)}/{needed}")

        print(f"✓ Generated {len(synthetic_texts)} synthetic samples in {attempts} attempts")

        syn_df = pd.DataFrame({
            "synthetic_text": synthetic_texts
        })

        print("\n--- SYNTHETIC SAMPLE OUTPUT (First 20) ---")
        print(syn_df.head(20).to_string())

        # ===============================================================
        # STEP 7: CALCULATE METRICS
        # ===============================================================
        print("\n" + "=" * 90)
        print("STEP 7: CALCULATE METRICS")
        print("=" * 90)

        def bleu_like(real, fake):
            r = set(real.split())
            f = set(fake.split())
            return len(r & f) / max(len(f), 1)

        def rouge_like(real, fake):
            r = set(real.split())
            f = set(fake.split())
            return len(r & f) / max(len(r), 1)

        def distinct1(texts):
            words = []
            for t in texts:
                words.extend(t.split())
            return len(set(words)) / max(len(words), 1)

        def distinct2(texts):
            bg = []
            for t in texts:
                w = t.split()
                for i in range(len(w) - 1):
                    bg.append((w[i], w[i + 1]))
            return len(set(bg)) / max(len(bg), 1)

        def self_bleu(texts, n=300):
            vals = []
            sample = texts[:n]

            for i in range(len(sample) - 1):
                vals.append(bleu_like(sample[i], sample[i + 1]))

            return np.mean(vals) if len(vals) > 0 else 0.0

        def privacy_leak(real, fake):
            rs = set(real)
            copied = sum(1 for x in fake if x in rs)
            return copied / max(len(fake), 1)

        # pairwise compare
        N = min(1000, len(real_texts), len(synthetic_texts))

        bleu_scores = []
        rouge_scores = []

        for i in range(N):
            bleu_scores.append(bleu_like(real_texts[i], synthetic_texts[i]))
            rouge_scores.append(rouge_like(real_texts[i], synthetic_texts[i]))

        BLEU = round(np.mean(bleu_scores), 4)
        ROUGE = round(np.mean(rouge_scores), 4)
        D1 = round(distinct1(synthetic_texts), 4)
        D2 = round(distinct2(synthetic_texts), 4)
        SELF = round(self_bleu(synthetic_texts), 4)
        PRIV = round(privacy_leak(real_texts, synthetic_texts), 4)

        print(f"BLEU Score          : {BLEU}")
        print(f"ROUGE Score         : {ROUGE}")
        print(f"Distinct-1          : {D1}")
        print(f"Distinct-2          : {D2}")
        print(f"Self-BLEU           : {SELF}")
        print(f"Privacy Leakage Rate: {PRIV}")

        # ===============================================================
        # STEP 8: SAVE FILES
        # ===============================================================
        print("\n" + "=" * 90)
        print("STEP 8: SAVE FILES")
        print("=" * 90)

        save_folder = "HYDRA"
        os.makedirs(save_folder, exist_ok=True)

        file_syn = os.path.join(save_folder, "HYDRAGAN_SYNTHETIC_TEXT_FINAL.csv")
        file_met = os.path.join(save_folder, "HYDRAGAN_METRICS_FINAL.csv")

        syn_df.to_csv(file_syn, index=False)
        print(f"✓ Saved: HYDRAGAN_SYNTHETIC_TEXT_FINAL.csv")

        # ✅ CREATE METRICS WITH PROPER ORDERING
        metrics_df = pd.DataFrame({
            "Metric": [
                "BLEU Score",
                "ROUGE Score",
                "Distinct-1",
                "Distinct-2",
                "Self-BLEU",
                "Privacy Leakage Rate"
            ],
            "Value": [
                BLEU,
                ROUGE,
                D1,
                D2,
                SELF,
                PRIV
            ]
        })

        metrics_df.to_csv(file_met, index=False)
        print(f"✓ Saved: HYDRAGAN_METRICS_FINAL.csv")

        # ===============================================================
        # STEP 9: READ FILES FOR DASHBOARD DISPLAY
        # ===============================================================
        print("\n" + "=" * 90)
        print("STEP 9: PREPARE DASHBOARD OUTPUT")
        print("=" * 90)

        # Read synthetic text
        synthetic_read = pd.read_csv(file_syn)
        print(f"\n✓ Reading HYDRAGAN_SYNTHETIC_TEXT_FINAL.csv")
        print(synthetic_read.head(10).to_string())

        # Read metrics
        metrics_read = pd.read_csv(file_met)
        print(f"\n✓ Reading HYDRAGAN_METRICS_FINAL.csv")
        print(metrics_read.to_string())

        # ===============================================================
        # FORMAT FOR DASHBOARD
        # ===============================================================

        # Synthetic Text Display
        synthetic_display = '\n\n'.join([
            f"{i+1}. {row['synthetic_text'][:100]}..."
            for i, (_, row) in enumerate(synthetic_read.head(10).iterrows())
        ])

        # Performance Metrics Display - ✅ KEEP AS DICTIONARY
        performance_dict = {}
        for idx, row in metrics_read.iterrows():
            metric_name = row['Metric']
            metric_value = float(row['Value'])
            performance_dict[metric_name] = metric_value

        # ===============================================================
        # PREPARE RESPONSE
        # ===============================================================

        response = {
            'status': 'success',
            'synthetic_text': synthetic_display,
            'performance': performance_dict,  # ✅ Dictionary format for displayHydraganMetrics
            'total_generated': len(synthetic_texts),
            'approved_input': len(real_texts)
        }

        print("\n" + "=" * 90)
        print("✓ BLOCK 3 COMPLETED SUCCESSFULLY")
        print("=" * 90 + "\n")

        return response

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return {
            'status': 'error',
            'error': str(e)
        }