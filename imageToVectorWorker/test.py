import unittest
from unittest.mock import MagicMock, patch
from Pipeline import Pipeline, ENCRIPTION_TYPE
import json


class TestPipeline(unittest.TestCase):

    @patch('Pipeline.redis.Redis')
    @patch('Pipeline.KafkaProducer')
    @patch('Pipeline.KafkaConsumer')
    @patch('Pipeline.Converter')
    def setUp(self, mock_converter, mock_consumer, mock_producer, mock_redis):
        # Mock Redis
        self.mock_redis_instance = MagicMock()
        mock_redis.return_value = self.mock_redis_instance

        # Mock Kafka producer and consumer
        self.mock_producer_instance = MagicMock()
        mock_producer.return_value = self.mock_producer_instance
        self.mock_consumer_instance = MagicMock()
        mock_consumer.return_value = self.mock_consumer_instance

        # Mock Converter
        self.mock_converter_instance = MagicMock()
        mock_converter.return_value = self.mock_converter_instance

        # Create Pipeline instance
        self.pipeline = Pipeline('Worker-1')

    def test_convert_data(self):
        """Test that _convert_data uses Converter correctly"""
        dummy_data = b"imagebytes"
        expected_keypoints = [{'x': 1, 'y': 2}]

        self.mock_converter_instance.point_detection.return_value = "detected_points"
        self.mock_converter_instance.extract_keypoints.return_value = expected_keypoints

        result = self.pipeline._convert_data(dummy_data)

        self.mock_converter_instance.point_detection.assert_called_once_with(dummy_data)
        self.mock_converter_instance.extract_keypoints.assert_called_once_with("detected_points")
        self.assertEqual(result, expected_keypoints)

    def test_produce_data(self):
        """Test that _produce_data retrieves data from Redis and sends to Kafka"""
        client_uuid = "client123"
        redis_list = [json_str := '{"x": 1, "y": 2}']
        self.mock_redis_instance.lrange.return_value = redis_list

        self.pipeline._produce_data(client_uuid)

        # Redis should have been read
        self.mock_redis_instance.lrange.assert_called_once_with(client_uuid, 0, -1)

        # Kafka producer should send message with correct key and value
        expected_payload = {'keypoints': [json.loads(json_str)]}
        self.mock_producer_instance.send.assert_called_once_with(
            'keypoints',
            value=expected_payload,
            key=client_uuid.encode(ENCRIPTION_TYPE)
        )
        self.mock_producer_instance.flush.assert_called_once()

    def test_start_consuming_skips_message_without_key(self):
        """Test that messages without keys are skipped"""
        mock_message = MagicMock()
        mock_message.key = None
        self.mock_consumer_instance.__iter__.return_value = [mock_message]

        # Ensure no redis or converter calls happen
        self.pipeline.start_consuming()

        self.mock_redis_instance.rpush.assert_not_called()
        self.mock_converter_instance.point_detection.assert_not_called()

    def test_start_consuming_processes_message(self):
        """Test normal message processing"""
        mock_message = MagicMock()
        mock_message.key = b'client123'
        mock_message.value = b'some_image_data'

        self.mock_converter_instance.point_detection.return_value = "points"
        self.mock_converter_instance.extract_keypoints.return_value = [{'x': 1, 'y': 2}]

        self.mock_consumer_instance.__iter__.return_value = [mock_message]
        self.pipeline.start_consuming()

        # Verify converter called
        self.mock_converter_instance.point_detection.assert_called_once()
        self.mock_converter_instance.extract_keypoints.assert_called_once()

        # Verify Redis got pushed data
        self.mock_redis_instance.rpush.assert_called_once()

    def test_start_consuming_stop_message(self):
        """Test that 'stop' message triggers _produce_data and redis deletion"""
        mock_message = MagicMock()
        mock_message.key = b'client123'
        mock_message.value = b'stop'

        self.mock_consumer_instance.__iter__.return_value = [mock_message]

        with patch.object(self.pipeline, '_produce_data') as mock_produce:
            self.pipeline.start_consuming()

            mock_produce.assert_called_once_with('client123')
            self.mock_redis_instance.delete.assert_called_once_with('client123')


if __name__ == '__main__':
    unittest.main()

