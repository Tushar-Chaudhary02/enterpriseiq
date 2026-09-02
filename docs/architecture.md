# EnterpriseIQ Architecture

## Business objective

EnterpriseIQ helps customer-retention teams identify customers who may churn,
understand the contributing factors, retrieve applicable company policies, and
produce evidence-backed retention recommendations.

## Planned system

```mermaid
flowchart TD
    UI["User interface"] --> API["FastAPI service"]
    API --> Agent["LangGraph agent"]
    Agent --> ML["Churn prediction tool"]
    Agent --> SQL["Read-only SQL tool"]
    Agent --> RAG["Policy retrieval tool"]
    ML --> Registry["MLflow registry"]
    SQL --> DB["PostgreSQL"]
    RAG --> Vector["PostgreSQL and pgvector"]
    Agent --> Approval["Human approval"]
    Agent --> Trace["LLM observability"]
```

## Major capabilities

1. Customer churn prediction
2. Model explanation
3. Natural-language customer analysis
4. Document retrieval with citations
5. Controlled SQL queries
6. Agent tool orchestration
7. Human approval for sensitive actions
8. ML and RAG evaluation
9. Monitoring and audit logging

## Architectural principles

- Reproducible data and ML pipelines
- Separation between training and inference
- Read-only access for AI-generated SQL
- No secrets committed to Git
- Structured and validated model output
- Citations for document-grounded answers
- Human approval before sensitive actions
- Automated testing before deployment