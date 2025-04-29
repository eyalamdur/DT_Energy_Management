import decision_transformer.decision_transformer as DecisionTransformer

def main():
    dt_model = DecisionTransformer(state_dim=6, act_dim=5, rtg_dim=1)
    print("works:)")

if __name__ == "__main__":
    main()