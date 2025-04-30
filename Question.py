from Query import query_rag  # Example usage

while True:
    question = input("Ask your question (or type 'e' to exit): ")
    if question.lower() == 'e':
        print("Exiting...")
        break

    result = query_rag(question)
    print("Answer:", result["answer"])
    print("\nSources:")
    for i, source in enumerate(result["sources"]):
        print(f"Source {i+1}: {source}")
