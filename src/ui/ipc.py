"""
UI i pipeline ce se pokretati kao odvojeni procesi.
UI salje JSON zahteve preko stdin, pipeline odgovara JSON-om preko stdout.
Protokol je jedan JSON objekat po liniji.
RecognitionResult i EvaluationResult ce se serijalizovati preko dataclasses.asdict().
IPC sloj ce se implementirati kad dodamo Flet UI.

ZASAD JE OVO SAMO KONCEPT. Zeleo bih da na kraju bude ovako:
poseban proces za UI (Flet) i poseban za pipeline, komunikacija preko JSON poruka.
Ali za sad pipeline direktno pozivamo iz CLI-ja.
"""
