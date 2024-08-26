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
INTO OUTFILE '/var/lib/mysql-files/data_chembl27_14_04_23.csv'
FROM activities AS act
JOIN assays AS ass ON act.assay_id = ass.assay_id
JOIN target_dictionary AS trgd ON ass.tid = trgd.tid
JOIN compound_structures AS cmpstc ON act.molregno = cmpstc.molregno
JOIN molecule_dictionary AS md ON act.molregno = md.molregno
JOIN target_components AS tc ON ass.tid = tc.tid
JOIN component_sequences AS cs ON tc.component_id = cs.component_id
;
