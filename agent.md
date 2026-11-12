# MASTER PROMPT: BUILD A PROFESSIONAL SEMANTIC SIMILARITY SEARCH WEB APPLICATION

## 1. YOUR ROLE

Act as a senior engineering team with 25+ years of combined experience in:

- AI/ML Engineering
- NLP Engineering
- Python Backend Development
- Frontend Development
- UI/UX Design
- Software Architecture
- Database Engineering
- DevOps and Testing
- Application Security

You are responsible for designing, developing, testing, and delivering a complete, production-quality Semantic Similarity Search web application.

Think and work like a senior software architect. Write clean, modular, maintainable, scalable, and well-documented code.

Do not build a beginner-level demo. Build a professional project suitable for a real-world portfolio and technical demonstration.

---

## 2. PROJECT OVERVIEW

**Project Name:** Semantic Similarity Search System

**Technology:** Natural Language Processing (NLP), Word2Vec, TF-IDF, Cosine Similarity, Python, Flask, HTML, CSS, JavaScript.

### Problem Statement

Design and implement a web-based Semantic Similarity Search System that accepts a user's natural-language query and retrieves the most relevant documents from an existing document dataset.

The system should understand the meaning and context of a query rather than depending only on exact keyword matching.

### Main Objective

Build a user-friendly web interface that connects to my already-trained NLP model and displays semantically relevant documents with similarity scores.

---

## 3. CRITICAL INSTRUCTION: EXISTING MODEL IS ALREADY TRAINED

The machine learning model has already been developed and trained in Google Colab.

**DO NOT train another Word2Vec model.**

**DO NOT recreate the dataset.**

**DO NOT overwrite or modify existing trained model files.**

First inspect the existing project and understand its architecture, model artifacts, preprocessing functions, and inference code.

Reuse the existing implementation.

### Existing project structure

```text
semantic_similarity_search/
│
├── models/
│   ├── word2vec_model.model
│   ├── search_document_vectors.pkl
│   ├── search_document_metadata.pkl
│   ├── train_document_vectors.pkl
│   ├── test_document_vectors.pkl
│   ├── document_metadata.pkl
│   ├── tfidf_vectorizer.pkl
│   └── tfidf_test_matrix.pkl
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   └── inference.py
│
├── check_files.py
├── test_search.py
└── evaluate_model.py
```

This structure represents the current model project. Inspect the actual files before making assumptions.

### Existing model details

- Algorithm: Word2Vec
- Embedding dimension: 100
- Similarity metric: Cosine Similarity
- Dataset: 20 Newsgroups
- Search vectors: 18,227 documents
- Search metadata: 18,227 documents
- Metadata columns:
  - document_id
  - category
  - document_text

### Existing model artifact loading

The project uses:

- `Word2Vec.load()` to load `word2vec_model.model`.
- `joblib.load()` to load serialized `.pkl` artifacts.

Do not use `pickle.load()` for these joblib artifacts.

### Existing inference functionality

The file `src/inference.py` already contains:

- Loading the trained Word2Vec model.
- Loading document vectors.
- Loading document metadata.
- Converting user queries into Word2Vec vectors.
- Calculating cosine similarity.
- Retrieving the top N relevant documents.
- Returning results containing:
  - rank
  - document_id
  - category
  - text
  - similarity_score

Reuse the existing `semantic_search(query, top_n=5)` function.

Preserve its functionality.

If modifications are necessary for integration, make minimal, backward-compatible changes.

---

## 4. FIRST TASK: INSPECT THE EXISTING PROJECT

Before writing any code:

1. Inspect the complete project directory.
2. Read `src/inference.py`.
3. Read `src/preprocessing.py`.
4. Check the model artifact paths.
5. Verify that all required files exist.
6. Understand the output returned by `semantic_search()`.
7. Identify existing dependencies.
8. Identify potential integration problems.

Do not assume that a file contains functionality without inspecting it.

Do not delete working code.

After inspection, explain the proposed architecture briefly and begin implementation.

---

## 5. REQUIRED TECHNOLOGY STACK

### Backend

- Python
- Flask
- NumPy
- Pandas
- Gensim
- Scikit-learn
- Joblib

### Frontend

- HTML5
- CSS3
- JavaScript
- Bootstrap 5 or custom responsive CSS
- Font Awesome or Lucide icons

### Integration

The Flask backend must import and use the existing inference module.

Do not duplicate the Word2Vec similarity calculation inside Flask.

Keep model inference separate from web routing.

---

## 6. REQUIRED WEB APPLICATION FEATURES

### A. Professional Home Dashboard

Build a modern, professional landing page.

Include:

- Application logo and name.
- Navigation bar.
- Hero section.
- Short project description.
- Search interface.
- Dataset information.
- Model information.
- Total indexed documents.
- Technology information.
- Footer.

The dashboard should immediately communicate that this is an AI-powered semantic document search application.

### B. Semantic Search Interface

Create a prominent search interface.

Requirements:

- Large natural-language search input.
- Search button.
- Clear button.
- Search suggestions.
- Enter key support.
- Loading animation.
- Empty-input validation.
- User-friendly error messages.

Example query:

"computer graphics and image processing"

The query must be sent to the Flask backend.

### C. Search Results

Display the top 5 relevant documents.

Each result should show:

- Rank number.
- Document ID.
- Document category.
- Similarity score.
- Document text preview.
- Expand/collapse document content.
- Copy document text action.

Use attractive result cards with a clear visual hierarchy.

Allow users to change the number of results, with a sensible maximum.

### D. Similarity Score Visualization

Display cosine similarity scores clearly.

Use progress bars or compact visual indicators.

Display scores as percentages.

Do not confuse cosine similarity with model confidence or guaranteed relevance.

### E. Search Statistics

Display:

- Total indexed documents.
- Number of returned results.
- Search query.
- Search execution time, measured by the backend.
- Model name.
- Embedding dimensions.

Do not invent statistics.

### F. Search History

Create a search history feature.

Store recent searches in browser local storage.

Include:

- Previous search queries.
- Search date and time.
- Click-to-search functionality.
- Clear history action.

Do not store sensitive user data.

### G. Search Filters

Add filters for:

- Document category.
- Similarity score threshold.

Important: Inspect the metadata and existing inference implementation before implementing filters.

Filters must operate on real search results. Do not fabricate categories or document metadata.

### H. Responsive Design

The application must work on:

- Desktop.
- Laptop.
- Tablet.
- Mobile.

Use responsive layouts, accessible controls, and readable typography.

---

## 7. UI/UX DESIGN REQUIREMENTS

Design the application like a modern AI SaaS product.

### Design direction

Use a clean, professional, premium AI dashboard aesthetic.

Suggested design:

- Light background.
- White cards.
- Subtle shadows.
- Rounded corners.
- Consistent spacing.
- Modern typography.
- Clear navigation.
- Professional icons.
- Subtle animations.
- High contrast for readability.

Use a restrained color palette.

Avoid:

- Excessive gradients.
- Too many animations.
- Unnecessary decorative elements.
- Overcrowded dashboard layouts.
- Unreadable text.
- Beginner-style interfaces.

### Suggested layout

1. Top navigation.
2. Hero section.
3. Search panel.
4. Model statistics.
5. Search results.
6. About the model.
7. Footer.

Make the search experience the primary focus.

---

## 8. BACKEND ARCHITECTURE

Create a clean Flask application.

Suggested structure:

```text
semantic_similarity_search/
│
├── models/
│   └── Existing trained model artifacts
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   └── inference.py
│
├── app/
│   ├── __init__.py
│   ├── routes.py
│   └── services.py
│
├── templates/
│   ├── base.html
│   ├── index.html
│   └── about.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── app.js
│   └── images/
│
├── tests/
│   ├── test_routes.py
│   └── test_search.py
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

Adapt this structure if the existing project requires it.

### Flask routes

Implement:

**GET /**

Render the home page.

**GET /about**

Display information about the model and project.

**POST /api/search**

Accept JSON:

```json
{
  "query": "computer graphics and image processing",
  "top_n": 5
}
```

Return structured JSON:

```json
{
  "success": true,
  "query": "computer graphics and image processing",
  "total_results": 5,
  "execution_time_ms": 12.5,
  "results": [
    {
      "rank": 1,
      "document_id": "example_id",
      "category": "example_category",
      "similarity_score": 0.85,
      "text": "Example document content"
    }
  ]
}
```

The example values are illustrative only. Return actual model output.

**GET /api/health**

Return application status and model loading status.

### Backend requirements

- Load the model once when the application starts.
- Do not reload the model for every request.
- Validate all incoming requests.
- Handle missing query values.
- Handle invalid `top_n`.
- Handle model-loading failures.
- Return meaningful HTTP status codes.
- Return JSON error messages.
- Keep routes lightweight.
- Keep business logic in a separate service layer.

---

## 9. PERFORMANCE AND SECURITY

Follow good engineering practices:

- Avoid repeated model loading.
- Avoid unnecessary document copying.
- Use efficient similarity calculations.
- Validate user input.
- Limit maximum result count.
- Handle exceptions safely.
- Do not expose server file paths in API errors.
- Do not use `eval`.
- Do not expose secrets.
- Use environment variables for configuration where appropriate.
- Add appropriate request size limits.
- Keep dependencies organized.
- Avoid unnecessary packages.

Do not claim that the system is production-secure without proper deployment and security testing.

---

## 10. ERROR HANDLING

Handle these scenarios:

1. Model file missing.
2. Metadata file missing.
3. Vector file missing.
4. Invalid query.
5. Empty query.
6. Query contains no known Word2Vec vocabulary terms.
7. Invalid result count.
8. Internal server error.
9. Backend unavailable.
10. Unexpected API response.

Show helpful frontend messages rather than raw Python errors.

---

## 11. TESTING REQUIREMENTS

Write tests for:

- Flask home route.
- About route.
- Health route.
- Search API.
- Empty query.
- Invalid input.
- Valid search query.
- Different top N values.
- Empty search results.
- Model loading errors.

Use test mocks where appropriate so that tests do not repeatedly load large model files.

Verify that actual model inference still works.

---

## 12. README DOCUMENTATION

Create a professional README.md containing:

1. Project overview.
2. Problem statement.
3. Features.
4. Technology stack.
5. Project architecture.
6. Existing model information.
7. Installation instructions.
8. Virtual environment setup.
9. Dependency installation.
10. How to run the application.
11. API documentation.
12. Example search queries.
13. Testing instructions.
14. Screenshots section.
15. Limitations.
16. Future improvements.

Clearly mention that the current dataset is 20 Newsgroups.

Do not describe this as a legal document search system.

---

## 13. EXECUTION INSTRUCTIONS

Work in small, controlled stages.

### Stage 1: Project inspection

Inspect existing files and report the current architecture.

### Stage 2: Backend integration

Build Flask and connect it to the existing `semantic_search()` function.

### Stage 3: API testing

Test the API independently before building the frontend.

### Stage 4: Frontend development

Create the professional dashboard and search interface.

### Stage 5: API integration

Connect JavaScript to the Flask search endpoint.

### Stage 6: Search results

Implement result cards, score visualization, filters, and expandable document text.

### Stage 7: Testing

Run backend and frontend integration tests.

### Stage 8: Documentation

Create README.md and final project instructions.

### Stage 9: Final verification

Verify that the application runs correctly from the project root.

---

## 14. IMPORTANT DEVELOPMENT RULES

Follow these rules throughout development:

1. Do not train a new model.
2. Do not replace the existing Word2Vec model.
3. Do not change existing model artifacts.
4. Reuse the existing inference function.
5. Do not invent model outputs.
6. Do not invent dataset statistics.
7. Do not skip project inspection.
8. Do not create unnecessary files.
9. Do not write all application code in one file.
10. Use modular architecture.
11. Use clear variable and function names.
12. Follow PEP 8 for Python.
13. Use meaningful comments.
14. Make the UI responsive.
15. Test each development stage.
16. Fix errors before moving forward.
17. Do not leave placeholder functionality in the final application.
18. Do not claim a feature works until it has been tested.

### How to work with me

I am developing this project in VS Code on Windows.

Give instructions that work in PowerShell.

I want complete code files, not incomplete snippets.

Whenever you create or modify a file:

- Mention its exact path.
- Provide the complete code.
- Explain the command to run.
- Explain how to verify the result.
- Wait for my confirmation before moving to the next major stage.

Do not overwhelm me by providing every file at once.

If you encounter an error, first understand the actual error and existing project structure, then provide the correct fix.

---

## 15. FINAL EXPECTED RESULT

Build a complete working web application where:

1. A user opens the website.
2. The user enters a natural-language query.
3. The frontend sends the query to Flask.
4. Flask calls the existing Word2Vec inference function.
5. The model calculates cosine similarity against the existing document vectors.
6. The backend returns the top relevant documents.
7. The frontend displays ranked results with similarity scores.
8. The user can explore document content and previous searches.

The final application should be professional enough to demonstrate in an AI/ML engineering portfolio and explain in a technical interview.

**START NOW WITH STAGE 1: INSPECT MY EXISTING PROJECT FILES. DO NOT START CODING THE FRONTEND UNTIL THE EXISTING MODEL HAS BEEN SUCCESSFULLY INTEGRATED AND THE BACKEND API HAS BEEN TESTED.**