"""eval/ — verified vs prose RAG comparison eval.

Modules:
  xbrl_fetcher    — pulls canonical financial facts from SEC EDGAR
  question_set    — generates 30 questions with typed ground truth
  scorer          — extracts numeric claims, scores against XBRL
  runner          — runs both pipelines against the question set
  report          — Wilson CIs, McNemar, per-bucket breakdown
"""