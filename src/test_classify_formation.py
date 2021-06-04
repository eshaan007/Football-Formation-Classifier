from utils.classify_formation import load_dataset, FormationClassification

pos, player_ids = load_dataset('prepped_tromso_stomsgodset_first.csv', 0, 100)
fc = FormationClassification(pos, player_ids)
fc.compute_form_summary()
fc.visualize_form_summary()
