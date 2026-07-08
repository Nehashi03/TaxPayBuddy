# 💰 TaxPayBuddy
## AI Multi-Agent Tax Assistant for Sri Lanka

TaxPayBuddy is an AI-powered Multi-Agent Tax Assistant designed to help users understand and navigate the Sri Lankan tax system. The project uses multiple specialized AI agents, where each agent is responsible for a specific tax-related domain. A Router Agent detects the user's intent and forwards requests to the appropriate tax agent.

---

# 👥 Team Members & Responsibilities

| Agent | Responsibility |
|--------|----------------|
| 🤖 Agent 1 | TIN Registration |
| 🤖 Agent 2 | Individual Income Tax |
| 🤖 Agent 3 | Corporate Income Tax |
| 🤖 Agent 4 | Withholding Tax (WHT) |
| 🤖 Agent 5 | Router Agent (Intent Detection & Scope Control) |

---

# 📂 Project Structure

```
TaxPayBuddy/
│
├── agents/        # AI agents implementation
├── data/          # Tax datasets and knowledge base
├── docs/          # Documentation
├── shared/        # Shared utilities and common modules
├── backend/       # Backend API and services
├── frontend/      # User Interface
└── README.md
```

---

# ⚙️ Workflow

The project follows a Git branching workflow to ensure smooth collaboration.

1. Clone the repository.
2. Create your own feature branch.
3. Work only on your assigned agent.
4. Commit your changes.
5. Push your branch to GitHub.
6. Create a Pull Request (PR).
7. After review, merge into the `develop` branch.
8. Once all features are tested, merge `develop` into `main`.

---

# 🌿 Branch Strategy

```
main
│
└── develop
      ├── feature/agent-1
      ├── feature/agent-2
      ├── feature/agent-3
      ├── feature/agent-4
      └── feature/router-agent
```

---

# 🎯 Project Goals

- Simplify Sri Lankan tax-related processes
- Provide AI-powered tax assistance
- Reduce confusion regarding tax regulations
- Deliver accurate responses using specialized AI agents
- Improve user experience through intelligent routing

---

# 🛠️ Technologies

- Python
- FastAPI
- LangChain / Multi-Agent Framework
- React.js
- Git & GitHub
- Vector Database (Optional)
- LLM APIs (OpenAI / Gemini)

---

# 🚀 Future Enhancements

- VAT Agent
- PAYE Tax Agent
- Tax Calculator
- Document Upload & Analysis
- Multilingual Support (English, Sinhala, Tamil)
- RAG-based Knowledge Retrieval

---

# 📌 Development Guidelines

- Work only on your assigned branch.
- Keep commits small and meaningful.
- Test your changes before creating a Pull Request.
- Resolve merge conflicts before merging.
- Follow project coding standards.

---

# 📄 License

This project is developed for educational purposes.
