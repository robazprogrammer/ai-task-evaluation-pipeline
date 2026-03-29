📊 AI-Augmented Real Estate Analytics Pipeline

A Real-World Task Design and Evaluation System for Human + AI Workflows

🧠 Overview

This project is an end-to-end analytics pipeline built on real-world residential real estate data. While originally designed for market analysis and forecasting, it is better understood as a task environment for evaluating how humans and AI systems perform on complex, ambiguous problems.

The system processes messy, incomplete data and produces structured outputs used for decision-making. It includes validation rules, domain constraints, and evaluation logic that mirror the kinds of challenges AI systems face in real-world applications.

This project reflects a broader research question:

How do humans and AI systems collaborate to solve unfamiliar, multi-step problems, and where do those systems break down?

🎯 Purpose

This pipeline serves three purposes:

Real-World Analytics Engine
Generate actionable insights from housing market data (sales trends, pricing, forecasting signals)
AI Task Environment
Simulate realistic, open-ended tasks that could be assigned to an AI agent (for example, analyze a housing market and identify trends)
Evaluation Framework
Provide structured outputs and validation rules that allow comparison between expected results and AI-generated outputs
⚙️ System Capabilities
Data Ingestion and Normalization
Handles multiple file encodings for real-world robustness
Standardizes inconsistent column naming conventions
Cleans and structures semi-structured data fields
Feature Engineering
Synthesizes domain-specific variables such as pool types and loan types
Integrates external economic data using Freddie Mac interest rates
Constructs derived metrics such as price reduction percentages and sale-to-list ratios
KPI Generation
Monthly and Year-to-Date analytics
Sales counts, median pricing, and appreciation trends
Price band segmentation and distribution analysis
Days on Market analysis across multiple dimensions
Validation and Constraints
KPI requirement checks to ensure required fields exist
Outlier filtering such as realistic sale-to-list ratio bounds
Data quality checks including missing values and duplicates
Visualization and Reporting
Automated chart generation using matplotlib
Excel report output with structured KPI tables
Clean, decision-ready outputs for business use
🧪 AI Task Design Layer

This system can be used to simulate tasks given to an AI model or agent.

Example Task

Analyze the housing market and identify the top-performing ZIP codes, pricing trends, and risk factors

What Makes This Difficult
Incomplete or inconsistent data
Multiple valid interpretations of the task
Need for aggregation, filtering, and reasoning
Domain-specific constraints
Why This Matters

Most AI systems perform well on narrow tasks but struggle when:

Data is messy
Instructions are ambiguous
Multiple steps must be chained together
Validation is required

This pipeline creates an environment where those limitations become visible.

⚠️ Observed AI Failure Modes

When applied to large language models, tasks like these often reveal consistent weaknesses:

Incorrect aggregation logic
Failure to apply constraints
Hallucinated insights not supported by data
Misinterpretation of domain concepts
Overconfidence in incomplete outputs

This system is designed to expose these failure points in a structured way.

🧠 Design Philosophy (AAEL Framework)

This project is grounded in the AI-Augmented Exploratory Learning (AAEL) framework:

Ask: Frame a problem or task
Adapt: Use AI to explore possible solutions
Analyze: Validate outputs and refine approach

Rather than treating AI as a tool for answers, this framework treats AI as a collaborative partner in problem-solving that requires human oversight and iteration.

📁 Project Structure
/real-estate-ai-pipeline
│
├── main_kpis.py              # Core pipeline and analytics engine
├── sample_data/
│   ├── sales_sample.csv      # Sample residential sales dataset
│   ├── freddiemac.csv        # Weekly mortgage rate data
├── out/
│   └── sample_outputs/       # Example charts and reports
├── notebooks/               # Optional analysis notebooks
├── README.md
📊 Required Data Files

This project uses two input datasets:

1. Sales Data

A residential real estate dataset containing fields such as:

Close of Escrow (COE)
Sold Price
Original List Price (OLP)
Final List Price (FLP)
City
Zip Code
Days on Market (DOM)

The repository includes a sample dataset for demonstration purposes.

2. Freddie Mac Interest Rates

A dataset containing:

Weekly mortgage rate observations
Columns: Week, Rate

This data is used to enrich the dataset with macroeconomic context.

Notes
The sales dataset is a subset of a larger MLS dataset
Data may be anonymized or reduced for demonstration
The pipeline is designed to scale to full production datasets
🚀 How to Run
python main_kpis.py
Outputs
Excel report with KPI tables
PNG charts for visualization
Structured outputs ready for analysis or presentation

All outputs are written to the out directory.

🔍 Why This Project Matters

This is not just an analytics tool.

It demonstrates a broader insight:

The gap between what AI can generate and what is required for reliable decision-making is still significant

Bridging that gap requires:

Structured validation systems
Domain-aware constraints
Iterative refinement
Human judgment

This project exists at that intersection.

🧠 About the Author

Robert Foreman
Doctoral Researcher in Educational Technology
MS in Business Analytics (4.0 GPA)

Creator of the AI-Augmented Exploratory Learning (AAEL) framework, focused on how AI enables individuals to perform beyond their current technical capabilities.

GitHub: https://github.com/robazprogrammer

⚡ Future Directions
AI-generated task evaluation layer for grading model outputs
Integration with LLM agents for automated analysis
Expansion into predictive modeling and simulation environments
Application to additional domains such as finance, education, and operations
