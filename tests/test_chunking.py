from tenderlens.chunking import chunk_text, make_chunks

def test_chunking_overlap():
    chunks = chunk_text("one two three four five six", chunk_size=4, overlap=1)
    assert chunks == ["one two three four", "four five six"]

def test_empty_text():
    assert chunk_text("   ") == []

def test_make_chunks_metadata():
    chunks = make_chunks([{"text": "a b c", "source": "tender.md", "page": 2}], chunk_size=2, overlap=0)
    assert chunks[0]["source"] == "tender.md"
    assert chunks[0]["page"] == 2
