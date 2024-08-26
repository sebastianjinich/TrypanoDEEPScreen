-- Filtro todos los compuestos cuyos ensayos son de SINGLE PROTEIN, son ensayos de binding, y poseen el standard type adecuado

SELECT act.molregno, md.chembl_id AS compund_chembl_id,
        trgd.chembl_id AS target_chemb_id,
        act.standard_relation,
        act.standard_value AS bioactivity,
        act.standard_units,
        act.standard_type,
        act.potential_duplicate,
        cmpstc.canonical_smiles AS smiles,
        bcs.sequence AS target_sequence
INTO OUTFILE '/var/lib/mysql-files/data_final.csv'
FROM activities AS act
JOIN (SELECT tid, assay_id, assay_type, chembl_id FROM assays) AS ass ON act.assay_id = ass.assay_id
JOIN target_dictionary AS trgd ON ass.tid = trgd.tid
JOIN compound_structures AS cmpstc ON act.molregno = cmpstc.molregno
JOIN molecule_dictionary AS md ON act.molregno = md.molregno
JOIN target_components AS tc ON ass.tid = tc.tid
JOIN bio_component_sequences AS bcs ON tc.component_id = bcs.component_id
WHERE (trgd.target_type = 'SINGLE PROTEIN')
    AND (ass.assay_type ='B')
    AND (act.standard_type IN ('IC50','EC50','AC50','Ki','Kd','Potency'))
    AND (act.pchembl_value IS NOT NULL)
LIMIT 10
;

-- QUERY posta!

SELECT act.molregno, 
    md.chembl_id AS comp_id,
    ass.tid as tid,
    trgd.chembl_id AS target,
    act.standard_relation AS relation,
    act.standard_value AS bioactivity,
    act.standard_units AS units,
    act.standard_type AS type,
    act.potential_duplicate,
    act.pchembl_value,
    trgd.organism,
    trgd.target_type,
    ass.assay_type,
    cmpstc.canonical_smiles AS smiles,
    cmpstc.standard_inchi_key AS inchi_key,
    cs.sequence AS sequence
INTO OUTFILE '/var/lib/mysql-files/data_chembl25_13_04_23.csv'
FROM activities AS act
JOIN assays AS ass ON act.assay_id = ass.assay_id
JOIN target_dictionary AS trgd ON ass.tid = trgd.tid
JOIN compound_structures AS cmpstc ON act.molregno = cmpstc.molregno
JOIN molecule_dictionary AS md ON act.molregno = md.molregno
JOIN target_components AS tc ON ass.tid = tc.tid
JOIN component_sequences AS cs ON tc.component_id = cs.component_id
;



-- query para las smiles

SELECT md.chembl_id, cmpstc.canonical_smiles
FROM compound_structures AS cmpstc 
JOIN molecule_dictionary AS md ON md.molregno = cmpstc.molregno
LIMIT 10;
