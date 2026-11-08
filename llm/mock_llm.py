def generate_answer(prompt):

    # Check if this is an evidence critic prompt
    if 'You are an Evidence Critic' in prompt or 'OUTPUT FORMAT (JSON only):' in prompt:
        # Return valid JSON for evidence critic
        return '''{
  "sufficient": true,
  "reason": "",
  "missing": [],
  "refined_query": ""
}'''
    else:
        return (
            "This is a mock response. "
            "The retrieved context was successfully passed "
            "to the LLM layer."
        )