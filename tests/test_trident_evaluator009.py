import base64
import math
from pathlib import Path
import sys
import types
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/submission009'))
from evaluate_trident import prepared_local_image,ndcg_at_10


class TridentEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='trident-path-test-')

    def tearDown(self):
        self.temporary.cleanup()

    def test_comma_and_space_filename_reaches_real_file(self):
        path=Path(self.temporary.name)/'comma, space.png'
        path.write_bytes(b'test-only-no-real-image')
        seen=[]
        class FakeImage:
            size=(32,64)
            def save(self,buffer,format):
                self.assert_png=format
                buffer.write(b'encoded-test-image')
        def process(conv,image_patch_size):
            item=conv[0]['content'][0]
            seen.append(item)
            self.assertTrue(Path(item['image']).is_file())
            self.assertNotIn('%2C',item['image'])
            self.assertEqual(image_patch_size,16)
            return [FakeImage()],None
        fake=types.ModuleType('qwen_vl_utils.vision_process');fake.process_vision_info=process
        with patch.dict(sys.modules,{'qwen_vl_utils.vision_process':fake}):
            encoded,size=prepared_local_image(path)
        self.assertEqual(base64.b64decode(encoded.split(',')[1]),b'encoded-test-image')
        self.assertEqual(size,[32,64])
        self.assertEqual(seen[0]['min_pixels'],262144)
        self.assertEqual(seen[0]['max_pixels'],1310720)

    def test_one_positive_and_miss(self):
        self.assertEqual(ndcg_at_10(['a','b'],['a']),1)
        self.assertEqual(ndcg_at_10(['x','y'],['a']),0)
        self.assertAlmostEqual(ndcg_at_10(['x','a'],['a']),1/math.log2(3))

    def test_multiple_positives_use_ideal_denominator(self):
        self.assertAlmostEqual(ndcg_at_10(['a','x','b'],['a','b']),(1+1/math.log2(4))/(1+1/math.log2(3)))

    def test_cutoff_is_ten(self):
        self.assertEqual(ndcg_at_10([str(i) for i in range(11)],['10']),0)

    def test_invalid_relevance_or_duplicate_rank_rejected(self):
        with self.assertRaises(ValueError):ndcg_at_10(['a'],[])
        with self.assertRaises(ValueError):ndcg_at_10(['a','a'],['a'])


if __name__=='__main__':unittest.main()
