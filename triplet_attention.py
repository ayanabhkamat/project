import unittest
import torch
from triplet_attention import TripletGlobalLocalAttention

class TestTripletGlobalLocalAttention(unittest.TestCase):
    def test_forward_shape(self):
        B, C, H, W = 2, 64, 32, 32
        L = H * W
        input_tensor = torch.randn(L, B, C)
        
        triplet_gl_att = TripletGlobalLocalAttention()
        output = triplet_gl_att(input_tensor, H, W)
        
        self.assertEqual(output.shape, (L, B, C))

    def test_forward_shape_image_input(self):
        B, C, H, W = 2, 64, 32, 32
        input_tensor = torch.randn(B, C, H, W)
        
        triplet_gl_att = TripletGlobalLocalAttention()
        output = triplet_gl_att(input_tensor, H, W)
        
        self.assertEqual(output.shape, (B, C, H, W))

if __name__ == '__main__':
    unittest.main()
