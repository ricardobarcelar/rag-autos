CREATE TABLE fila_rag (
    id serial, 
    data_hora timestamp DEFAULT now(), 
    referencia varchar(200), 
    documento varchar(200),
    tipo char(1),
    acao char(1), 
    conteudo text, 
    data_hora_processamento timestamp);

    COMMENT ON COLUMN fila_rag.tipo IS 'B - Binário, E - Estruturado';
    COMMENT ON COLUMN fila_rag.acao IS 'I - Inclusão, E - Exclusão';
    COMMENT ON COLUMN fila_rag.conteudo IS 'Caso tipo = B, o conteúdo deve ser um JSON contendo Hash e bucket';


CREATE INDEX idx_data_hora_processamento
    ON public.fila_rag USING btree
    (data_hora_processamento ASC NULLS FIRST);