-- Similarity search over document_sections, filtered by the user's role-based access.
-- Uses the HNSW index (approximate, fast) instead of an exact filter-then-sort.
-- hnsw.iterative_scan keeps expanding the graph search until enough authorized
-- rows satisfy p_limit, so the authorization filter doesn't silently starve the
-- result set the way a plain post-filtered ANN search could.
CREATE OR REPLACE FUNCTION search_document_sections(
  p_user_id UUID,
  p_query_embedding VECTOR(384),
  p_limit INT DEFAULT 5
)
RETURNS TABLE (id UUID, document_id UUID, content TEXT, distance FLOAT)
LANGUAGE plpgsql
AS $$
BEGIN
  SET LOCAL hnsw.iterative_scan = relaxed_order;
  RETURN QUERY
  SELECT ds.id, ds.document_id, ds.content,
         ds.embedding <=> p_query_embedding AS distance
  FROM document_sections ds
  WHERE ds.document_id IN (
    SELECT da.document_id
    FROM document_access da
    JOIN user_roles ur ON ur.role_id = da.role_id
    WHERE ur.user_id = p_user_id
  )
  ORDER BY ds.embedding <=> p_query_embedding
  LIMIT p_limit;
END;
$$;
