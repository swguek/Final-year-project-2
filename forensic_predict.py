import os
import re
import joblib
import pandas as pd
import numpy as np

# ==========================================
# PHASE 4: AUTOMATED FORENSIC ARTIFACT DETECTOR
# ==========================================

def load_ai_system():
    print("--- Loading Trained Random Forest System Artifacts ---")
    if not os.path.exists("phishing_rf_model.pkl") or not os.path.exists("tfidf_vectorizer.pkl"):
        raise FileNotFoundError("[ERROR] AI model assets missing! Please run main.py first to train the model.")
        
    model = joblib.load("phishing_rf_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    print("AI Forensic Brain Loaded Successfully!\n")
    return model, vectorizer

def extract_live_features(eml_text):
    text_lower = str(eml_text).lower()
    
    def check_auth(pattern):
        match = re.search(pattern, text_lower)
        if match and 'pass' in match.group(0):
            return 1
        return 0

    spf = check_auth(r'spf[:\s\-=]+[a-z]+')
    dkim = check_auth(r'dkim[:\s\-=]+[a-z]+')
    dmarc = check_auth(r'dmarc[:\s\-=]+[a-z]+')
    return [spf, dkim, dmarc]

if __name__ == "__main__":
    TARGET_FOLDER = "evaluation_set"
    
    rf_model, tfidf_vectorizer = load_ai_system()
    
    if not os.path.exists(TARGET_FOLDER):
        print(f"[ERROR] Target directory '{TARGET_FOLDER}' not found.")
        exit()
        
    eml_files = [f for f in os.listdir(TARGET_FOLDER) if f.endswith('.eml')]
    print(f"Found {len(eml_files)} evaluation email artifacts in custody.\n")
    

    forensic_reports = []
    
    print("Executing Real-time AI Classification on Evidence Pool...")
    print("-" * 80)
    

    for file_name in eml_files:
        file_path = os.path.join(TARGET_FOLDER, file_name)
        
        
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_content = f.read()
            
        
        text_features_sparse = tfidf_vectorizer.transform([raw_content])
          
        metadata_bits = extract_live_features(raw_content)
        
        from scipy.sparse import hstack
        X_live = hstack([text_features_sparse, np.array([metadata_bits])]).tocsr()
        
        prob_phishing = rf_model.predict_proba(X_live)[0][1]
        
        prediction = 1 if prob_phishing >= 0.30 else 0
        
        actual_label = int(file_name.split("_")[1].replace("label", ""))
        
        status = "MATCH" if prediction == actual_label else "MISMATCH (Error)"
        
        forensic_reports.append({
            "File Name": file_name,
            "Ground Truth": "Phishing (1)" if actual_label == 1 else "Ham (0)",
            "AI Prediction": "Phishing (1)" if prediction == 1 else "Ham (0)",
            "Phish Probability": f"{prob_phishing * 100:.1f}%",
            "Audit Status": status
        })

    df_report = pd.DataFrame(forensic_reports)
    pd.set_option('display.max_rows', None) 
    pd.set_option('display.width', 1000)
    
    print("\n" + "="*90)
    print(">>> PHASE 4 FINAL FORENSIC COMPARATIVE ANALYSIS REPORT <<<")
    print("="*90)
    print(df_report.to_string(index=False))
    print("="*90)
    
    matches = sum(df_report["Audit Status"] == "MATCH")
    print(f"\n[FINAL RESULT] AI System Blind Test Accuracy on Evidence Pool: {matches}/{len(eml_files)} ({matches/len(eml_files)*100:.2f}%)")
