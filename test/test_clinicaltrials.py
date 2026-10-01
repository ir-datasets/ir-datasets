import unittest
from ir_datasets.datasets.clinicaltrials import ClinicalTrialsDocs


XML = b"""<clinical_study>
  <id_info><nct_id>NCT00000000</nct_id></id_info>
  <brief_title>Brief</brief_title>
  <official_title>Official</official_title>
  <condition>Flu</condition>
  <brief_summary><textblock>Summary</textblock></brief_summary>
</clinical_study>"""


class TestClinicalTrialsParsing(unittest.TestCase):
    def test_leaf_elements(self):
        doc = ClinicalTrialsDocs('test', [])._parse_doc(XML)
        self.assertEqual(doc.doc_id, 'NCT00000000')
        self.assertEqual(doc.title, 'Official')
        self.assertEqual(doc.condition, 'Flu')
        self.assertEqual(doc.summary, 'Summary')
        self.assertEqual(doc.detailed_description, '')
        self.assertEqual(doc.eligibility, '')

    def test_brief_title_fallback(self):
        xml = XML.replace(b'<official_title>Official</official_title>', b'')
        self.assertEqual(ClinicalTrialsDocs('test', [])._parse_doc(xml).title, 'Brief')


if __name__ == '__main__':
    unittest.main()
