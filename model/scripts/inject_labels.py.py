import pandas as pd

def simulate_repayment_label(income, literacy, distance, sector):
    score = income * 0.00002 + literacy * 0.1 - distance * 0.1
    if sector == "agriculture":  #  df.columns_sum(sector -> x,y : x==y ? sum()) = avg()
        score -= 0.2
    elif sector == "it":
        score += 0.3
    return max(0, min(4, int(score)))  # clamp to [0, 4]

df = pd.read_csv("D:\Python\Credit_UnderWriting_Model\data\processed\synthetic_person_data.csv")

df["repayment_category"] = df.apply(
    lambda row: simulate_repayment_label(
        income=row["predicted_income"],
        literacy=row["female_literacy"],
        distance=row["distance_from_city"],
        sector=row["work_sector"]
    ),
    axis=1
)

df.to_csv("D:\Python\Credit_UnderWriting_Model\data\processed\user_income_features_with_labels.csv", index=False)
print("✅ repayment_category labels saved.")
