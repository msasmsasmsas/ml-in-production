#!/usr/bin/env python3

"""
Integration tests for Triton Inference Server deployment.
These tests require a running Triton server to pass.
"""

import unittest
import numpy as np
import tritonclient.http as httpclient
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestTritonIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Connect to Triton server
        cls.client = httpclient.InferenceServerClient(url='localhost:8000')

        # Wait for server to be ready (with timeout)
        server_ready = False
        timeout = 30  # seconds
        start_time = time.time()

        while not server_ready and (time.time() - start_time) < timeout:
            try:
                server_ready = cls.client.is_server_ready()
                if server_ready:
                    break
            except Exception:
                pass

            time.sleep(1)

        if not server_ready:
            raise RuntimeError("Triton server is not ready. Please ensure it's running.")

    def test_server_metadata(self):
        # Check server metadata
        metadata = self.client.get_server_metadata()
        self.assertIn('name', metadata)
        self.assertEqual(metadata['name'], 'triton')

    def test_model_ready(self):
        # Check if model is ready
        model_name = 'resnet50'
        is_ready = self.client.is_model_ready(model_name)
        self.assertTrue(is_ready, f"Model {model_name} is not ready")

    def test_model_metadata(self):
        # Check model metadata
        model_name = 'resnet50'
        metadata = self.client.get_model_metadata(model_name)

        # Verify inputs
        self.assertIn('inputs', metadata)
        self.assertEqual(len(metadata['inputs']), 1)
        self.assertEqual(metadata['inputs'][0]['name'], 'input')

        # Verify outputs
        self.assertIn('outputs', metadata)
        self.assertEqual(len(metadata['outputs']), 1)
        self.assertEqual(metadata['outputs'][0]['name'], 'output')

    def test_inference(self):
        # Test inference with dummy data
        model_name = 'resnet50'

        # Create dummy input data (batch of 1 image)
        input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)

        # Create input tensor
        inputs = []
        inputs.append(httpclient.InferInput('input', input_data.shape, "FP32"))
        inputs[0].set_data_from_numpy(input_data)

        # Create output tensor
        outputs = []
        outputs.append(httpclient.InferRequestedOutput('output'))

        # Send the inference request
        results = self.client.infer(model_name, inputs, outputs=outputs)

        # Verify the results
        output_data = results.as_numpy('output')

        # Check output shape (batch size, num classes)
        self.assertEqual(output_data.shape, (1, 1000))

        # Check output is normalized (sum of softmax probabilities should be close to 1)
        self.assertAlmostEqual(np.sum(np.exp(output_data[0]) / np.sum(np.exp(output_data[0]))), 1.0, places=5)

if __name__ == '__main__':
    unittest.main()
