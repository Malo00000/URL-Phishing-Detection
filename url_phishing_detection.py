# url_phishing_detection.py

import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Fonction pour extraire des caractéristiques d'une URL
def extract_features(url):
    features = {}
    features['url_length'] = len(url)
    features['num_digits'] = sum(c.isdigit() for c in url)
    features['num_subdomains'] = url.count('.') - 1
    features['contains_login'] = 1 if "login" in url else 0
    features['contains_secure'] = 1 if "secure" in url else 0
    features['contains_bank'] = 1 if "bank" in url else 0
    return features

# Exemples d'URLs (1 = phishing, 0 = safe)
urls = [
    "https://secure-login.bank.com", "http://example.com", 
    "https://randomsite.com/login", "http://dangerous.banksecure.org",
    "http://trustworthybank.com", "https://bank-login.com", 
    "http://my-secure-site.org", "http://safe-example.com"
]
labels = [1, 0, 1, 1, 0, 1, 0, 0]  # 1 = phishing, 0 = safe

# Création du DataFrame avec les URLs et les labels
data = pd.DataFrame(urls, columns=['url'])
data['label'] = labels

# Extraction des caractéristiques pour chaque URL
features = data['url'].apply(extract_features)
features_df = pd.DataFrame(features.tolist())
data = pd.concat([data, features_df], axis=1)

# Préparation des données pour le modèle
X = data[['url_length', 'num_digits', 'num_subdomains', 'contains_login', 'contains_secure', 'contains_bank']]
y = data['label']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entraînement du modèle
model = LogisticRegression()
model.fit(X_train, y_train)

# Prédictions et évaluation du modèle
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Tester avec de nouvelles URLs
test_urls = ["https://phishing-login.com", "http://my-safe-website.com"]
test_features = pd.DataFrame([extract_features(url) for url in test_urls])
predictions = model.predict(test_features)

print("\nTest URLs Predictions:")
for url, pred in zip(test_urls, predictions):
    print(f"{url}: {'Phishing' if pred == 1 else 'Safe'}")
