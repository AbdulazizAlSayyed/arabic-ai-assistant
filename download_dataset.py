import os
import pandas as pd
import requests

print("="*60)
print("Downloading Arabic Sentiment Dataset")
print("="*60)

# Create directories
os.makedirs("data/sentiment", exist_ok=True)

# Try to download from multiple sources
urls = [
    "https://raw.githubusercontent.com/bakrianoo/ArSAS/master/ArSAS-v2.tsv",
    "https://raw.githubusercontent.com/elnagara/HARD-Arabic-Dataset/master/data/astd-train.csv"
]

for url in urls:
    try:
        print(f"\nTrying: {url}")
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            filename = url.split("/")[-1]
            filepath = f"data/sentiment/{filename}"
            with open(filepath, "wb") as f:
                f.write(response.content)
            print(f"✅ Downloaded: {filename}")
        else:
            print(f"❌ Failed: {url}")
    except Exception as e:
        print(f"❌ Error: {e}")

# Create sample dataset if downloads failed
print("\n" + "="*60)
print("Creating sample dataset for immediate testing...")
print("="*60)

sample_data = """text,label
هذا المنتج ممتاز جدا,positive
الخدمة كانت سيئة للغاية,negative
أنا أحب هذا التطبيق,positive
لا يعجبني هذا,negative
جيد جدا,positive
سيء جدا,negative
رائع,positive
مشكلة كبيرة,negative
شكرا,positive
فاشل,negative
جيد نوعا ما,neutral
لا بأس,neutral
ممتاز,positive
سعر غالي,negative
سرعة ممتازة,positive
بطيء جدا,negative
خدمة رائعة,positive
مضيعة للوقت,negative
"""

sample_path = "data/sentiment/sentiment.csv"
with open(sample_path, "w", encoding="utf-8") as f:
    f.write(sample_data)

print(f"✅ Created sample dataset at: {sample_path}")
print(f"   Contains {len(sample_data.splitlines())-1} examples")

print("\n" + "="*60)
print("Setup Complete!")
print("="*60)