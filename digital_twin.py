import pandas as pd

df = pd.read_csv("data/factory_data.csv")

print("=" * 60)
print("FACTORY DIGITAL TWIN")
print("=" * 60)

production_rate = 100
hourly_cost = 500
downtime_cost = 2000

scenarios = {
    "Continue Operation": {
        "downtime": 0,
        "risk": 0.78
    },
    "Stop for Maintenance": {
        "downtime": 4,
        "risk": 0.15
    },
    "Reduce Machine Load": {
        "downtime": 1,
        "risk": 0.35
    }
}

results = []

for name, values in scenarios.items():

    downtime = values["downtime"]
    risk = values["risk"]

    production_loss = production_rate * downtime
    downtime_cost_total = downtime * downtime_cost
    risk_cost = risk * 5000

    total_cost = (
        downtime_cost_total +
        risk_cost
    )

    results.append({
        "Scenario": name,
        "Downtime Hours": downtime,
        "Failure Risk": risk,
        "Production Loss": production_loss,
        "Estimated Cost": round(total_cost, 2)
    })

result_df = pd.DataFrame(results)

print("\nWHAT-IF RESULTS\n")
print(result_df.to_string(index=False))

best = result_df.loc[
    result_df["Estimated Cost"].idxmin()
]

print("\n" + "=" * 60)
print("RECOMMENDED SCENARIO")
print("=" * 60)

print(best["Scenario"])
print("Estimated Cost:", best["Estimated Cost"])
print("Failure Risk:", best["Failure Risk"])
