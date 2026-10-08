import unittest

from serving_campaign import extra_arm_settings


class CombinationSettingsTests(unittest.TestCase):
    control = {'SPEC_TOKENS': '13', 'MI210_QUERY_TILE': '10', 'MI210_INT8_LM_HEAD': '0'}

    def test_combination_keeps_unchanged_control_settings(self):
        result = extra_arm_settings({'combo': {'MI210_NATIVE_GDN': '1',
                                    'MI210_INT8_QK': '1', 'SPEC_TOKENS': '9'}}, self.control, {})
        self.assertEqual(result['combo']['MI210_QUERY_TILE'], '10')
        self.assertEqual(result['combo']['MI210_INT8_QK'], '1')
        self.assertEqual(self.control['SPEC_TOKENS'], '13')

    def test_precision_model_and_unqualified_geometry_cannot_be_changed(self):
        for settings in ({'KV_DTYPE': 'bfloat16'}, {'MODEL_ROOT': '/other'},
                         {'SPEC_TOKENS': '16', 'MI210_NATIVE_GDN': '1'},
                         {'MI210_QUERY_TILE': '32'}, {'MI210_INT8_QK': True}):
            with self.assertRaises(ValueError):
                extra_arm_settings({'combo': settings}, self.control, {})

    def test_existing_or_path_labels_fail_without_mutating_control(self):
        for label in ('../old', 'old', 'bad label'):
            with self.assertRaises(ValueError):
                extra_arm_settings({label: {'SPEC_TOKENS': '7'}}, self.control, {'old': {}})
        self.assertEqual(self.control['SPEC_TOKENS'], '13')


if __name__ == '__main__':
    unittest.main()
