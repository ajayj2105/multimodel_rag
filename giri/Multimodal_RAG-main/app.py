import time

import streamlit as st
from pypdf import PdfReader

from rag.chunking import chunk_text
from rag.embeddings import get_jina_embeddings
from rag.llm import ask_llm
from rag.ocr import extract_text_from_image_bytes
from rag.reranker import simple_rerank
from rag.retriever import FAISSRetriever
from rag.vision import describe_image


st.set_page_config(page_title="Multimodal RAG Assistant", layout="wide")

st.markdown(
    """
    <style>
    body {
        font-family: "Inter", sans-serif;
    }
    .main-title {
        font-size: 40px;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0px;
    }
    .subtitle {
        font-size: 16px;
        color: #6b7280;
        margin-top: 4px;
        margin-bottom: 25px;
    }
    .panel {
        padding: 18px;
        border-radius: 14px;
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        margin-bottom: 18px;
    }
    .section-header {
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 12px;
        color: #111827;
    }
    .stButton button {
        border-radius: 10px;
        padding: 10px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='main-title'>Enterprise Multimodal RAG</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Retrieval-Augmented Generation over documents and images using Jina Embeddings and Groq Vision</div>",
    unsafe_allow_html=True,
)


if "history" not in st.session_state:
    st.session_state.history = []
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""

with st.sidebar:
    st.header("Configuration")

    groq_key = st.text_input("Groq API Key", type="password")
    jina_key = st.text_input("Jina API Key", type="password")

    model = st.selectbox("LLM Model", ["llama-3.1-8b-instant", "openai/gpt-oss-120b"])

    filter_type = st.radio("Retrieval Scope", ["all", "text", "image"], horizontal=True)
    use_image_ocr = st.checkbox("Use OCR on uploaded image", value=True)
    show_sources = st.checkbox("Show retrieval details", value=True)

    st.divider()
    st.subheader("Retrieval Tuning")
    chunk_size = st.slider("Chunk size (words)", 200, 1000, 400, 50)
    chunk_overlap = st.slider("Chunk overlap (words)", 0, 300, 80, 10)
    top_k = st.slider("Top-K retrieval", 1, 10, 5, 1)
    context_k = st.slider("Chunks in final context", 1, 5, 3, 1)

    st.divider()
    if st.button("Clear chat history", use_container_width=True):
        st.session_state.history = []
        st.session_state.last_answer = ""


col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Document Upload</div>", unsafe_allow_html=True)
    txt_file = st.file_uploader("Upload TXT or PDF", type=["txt", "pdf"])
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Image Upload</div>", unsafe_allow_html=True)
    img_file = st.file_uploader("Upload PNG or JPG", type=["png", "jpg", "jpeg"])
    st.markdown("</div>", unsafe_allow_html=True)


if (txt_file or img_file) and jina_key:
    with st.spinner("Processing knowledge sources..."):
        chunks = []
        metadata = []

        if txt_file:
            raw_text = ""
            if txt_file.name.endswith(".pdf"):
                try:
                    reader = PdfReader(txt_file)
                    raw_text = "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
                except Exception as exc:
                    st.error(f"Failed to parse PDF: {exc}")
                    st.stop()
            else:
                raw_text = txt_file.read().decode("utf-8", errors="ignore")

            safe_overlap = min(chunk_overlap, max(0, chunk_size - 1))
            text_chunks = chunk_text(raw_text, chunk_size=chunk_size, overlap=safe_overlap) if raw_text.strip() else []
            chunks.extend(text_chunks)
            metadata.extend(
                [
                    {"type": "text", "source": txt_file.name, "channel": "document", "chunk_id": i + 1}
                    for i, _ in enumerate(text_chunks)
                ]
            )

        if img_file:
            image_bytes = img_file.read()
            vision_text = ""
            if groq_key:
                try:
                    vision_text = describe_image(image_bytes, groq_key)
                except Exception as exc:
                    st.warning(f"Image description failed: {exc}")
            else:
                st.info("Groq key not provided. Skipping image description and using OCR only.")

            if vision_text:
                chunks.append("Image description: " + vision_text)
                metadata.append({"type": "image", "source": img_file.name, "channel": "vision", "chunk_id": 1})

            if use_image_ocr:
                ocr_text = ""
                try:
                    ocr_text = extract_text_from_image_bytes(
                        image_bytes, suffix="." + img_file.name.split(".")[-1].lower()
                    )
                except Exception as exc:
                    st.warning(f"OCR extraction failed: {exc}")

                if ocr_text:
                    chunks.append("Image OCR text: " + ocr_text)
                    metadata.append({"type": "image", "source": img_file.name, "channel": "ocr", "chunk_id": 1})

        if not chunks:
            st.error("No extractable content found in the uploaded document/image.")
            st.stop()

        try:
            embeddings = get_jina_embeddings(chunks, jina_key)
        except Exception as exc:
            st.error(f"Embedding generation failed: {exc}")
            st.stop()

        retriever = FAISSRetriever(embeddings, metadata)

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Query Interface</div>", unsafe_allow_html=True)
    query = st.text_input("Enter your question", placeholder="Example: What does the uploaded image explain?")
    run = st.button("Run Retrieval and Generate Answer", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if run:
        if not query.strip():
            st.error("Enter a query to continue.")
            st.stop()
        if not groq_key:
            st.error("Groq API Key is required to generate an answer.")
            st.stop()

        start = time.time()
        try:
            query_emb = get_jina_embeddings([query], jina_key)
        except Exception as exc:
            st.error(f"Query embedding generation failed: {exc}")
            st.stop()

        scope = None if filter_type == "all" else filter_type
        scored_results = retriever.search_with_scores(query_emb, top_k=top_k, filter_type=scope)

        retrieved_docs = [chunks[idx] for idx, _ in scored_results]
        reranked_docs = simple_rerank(query, retrieved_docs)
        context = "\n\n".join(reranked_docs[:context_k]) if reranked_docs else ""

        if context:
            try:
                answer = ask_llm(context, query, groq_key, model)
            except Exception as exc:
                st.error(f"LLM request failed: {exc}")
                st.stop()
        else:
            answer = "No matching context found for the selected retrieval scope."

        latency = round(time.time() - start, 2)
        st.session_state.last_answer = answer
        st.session_state.history.append({"query": query, "answer": answer, "latency": latency})

        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Answer</div>", unsafe_allow_html=True)
        st.write(answer)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Performance</div>", unsafe_allow_html=True)
        st.metric("Latency (seconds)", latency)
        st.markdown("</div>", unsafe_allow_html=True)

        if show_sources:
            with st.expander("Retrieved Context and Sources"):
                if scored_results:
                    for rank, (idx, dist) in enumerate(scored_results, start=1):
                        meta = metadata[idx]
                        st.markdown(
                            f"{rank}. type={meta.get('type')} | source={meta.get('source')} | "
                            f"channel={meta.get('channel')} | distance={dist:.4f}"
                        )
                        st.caption(chunks[idx][:280] + ("..." if len(chunks[idx]) > 280 else ""))
                else:
                    st.write("No retrieved items for the selected scope.")

        with st.expander("Final Context Sent to LLM"):
            st.text(context)

        st.download_button(
            label="Download Answer",
            data=st.session_state.last_answer,
            file_name="rag_answer.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with st.expander("Recent Chat History"):
        if st.session_state.history:
            for item in st.session_state.history[-5:]:
                st.markdown(f"Question: {item['query']}")
                st.markdown(f"Answer: {item['answer']}")
                st.markdown(f"Latency: {item['latency']}s")
                st.divider()
        else:
            st.write("No history yet.")
else:
    st.info("Upload a document or image and provide required API keys to begin.")
