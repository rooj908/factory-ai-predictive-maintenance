import json
from datetime import datetime


def human_supervisor_decision(ai_recommendation):

    print("\n" + "=" * 60)
    print("HUMAN SUPERVISOR REVIEW")
    print("=" * 60)

    print("\nAI Recommendation:")
    print(ai_recommendation)

    print("\nAvailable decisions:")
    print("1. APPROVE")
    print("2. REJECT")
    print("3. MODIFY")

    while True:

        choice = input("\nSelect decision (1/2/3): ").strip()

        if choice == "1":
            decision = "APPROVE"
            final_action = ai_recommendation
            break

        elif choice == "2":
            decision = "REJECT"

            reason = input(
                "Reason for rejection: "
            ).strip()

            final_action = "AI recommendation rejected"

            break

        elif choice == "3":
            decision = "MODIFY"

            final_action = input(
                "Enter modified action: "
            ).strip()

            reason = input(
                "Reason for modification: "
            ).strip()

            break

        else:
            print("Please select 1, 2 or 3.")

    record = {
        "timestamp": datetime.now().isoformat(),
        "ai_recommendation": ai_recommendation,
        "human_decision": decision,
        "final_action": final_action,
        "reason": locals().get("reason", "")
    }

    with open(
        "reports/human_decisions.json",
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            json.dumps(record) + "\n"
        )

    print("\nDecision recorded successfully.")

    return record


if __name__ == "__main__":

    result = human_supervisor_decision(
        "Inspect machine before returning it to full production."
    )

    print("\nFINAL HUMAN DECISION:")
    print(result)
