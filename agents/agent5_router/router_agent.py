def agent1_tin_registration(query):
    return f"Redirected to [Agent 1 - TIN Registration]. (Your query: '{query}')"

def agent2_individual_tax(query):
    return f"Redirected to [Agent 2 - Individual Tax]. (Your query: '{query}')"

def agent3_corporate_tax(query):
    return f"Redirected to [Agent 3 - Corporate Tax]. (Your query: '{query}')"

def agent4_withholding_tax(query):
    return f"Redirected to [Agent 4 - WHT]. (Your query: '{query}')"

def router_agent(user_query):
    query_lower = user_query.lower()
    
    tin_keywords = ["tin", "tax registration", "register tin", "get tin"]
    individual_keywords = ["personal", "individual", "salary", "paye", "personal tax"]
    corporate_keywords = ["company", "business", "corporate", "profit tax", "company tax"]
    wht_keywords = ["wht", "withholding", "withholding tax", "retention"]
    
    if any(keyword in query_lower for keyword in tin_keywords):
        return agent1_tin_registration(user_query)
    elif any(keyword in query_lower for keyword in individual_keywords):
        return agent2_individual_tax(user_query)
    elif any(keyword in query_lower for keyword in corporate_keywords):
        return agent3_corporate_tax(user_query)
    elif any(keyword in query_lower for keyword in wht_keywords):
        return agent4_withholding_tax(user_query)
    else:
        return "Sorry, your query is out of scope for the TaxPayBuddy system. Please ask a tax-related question."

if __name__ == "__main__":
    print("--- TaxPayBuddy Router Testing ---")
    questions = [
        "How do I register for a new TIN number?",
        "How much tax will be deducted from my salary?",
        "We need to calculate our company profit tax for 2025",
        "What is the WHT rate for rent?",
        "How is the weather today in Colombo?"
    ]
    for q in questions:
        print(f"\nUser: {q}")
        response = router_agent(q)
        print(f"System: {response}")