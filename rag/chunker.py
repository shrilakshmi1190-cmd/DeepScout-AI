import re


def chunk_text(text, chunk_size=1000, overlap=200):

    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:

        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "

        else:
            chunks.append(current_chunk.strip())

            overlap_text = current_chunk[-overlap:]

            current_chunk = overlap_text + " " + sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks



def chunk_pages(pages, chunk_size=1000, overlap=200):

    all_chunks = []

    for page_data in pages:

        text = page_data["text"]
        page_number = page_data["page"]
        source = page_data["source"]

        sentences = re.split(r'(?<=[.!?])\s+', text)

        current_chunk = ""

        for sentence in sentences:

            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + " "

            else:

                if current_chunk.strip():
                    all_chunks.append({
                        "text": current_chunk.strip(),
                        "page": page_number,
                        "source": source
                    })

                overlap_text = current_chunk[-overlap:]

                current_chunk = (
                    overlap_text + " " + sentence
                )

        if current_chunk.strip():
            all_chunks.append({
                "text": current_chunk.strip(),
                "page": page_number,
                "source": source
            })

    return all_chunks