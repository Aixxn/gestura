from kafka import KafkaProducer, KafkaConsumer
from Converter import Converter
import threading
import redis
import socket
import json

KAFKA_PORT = 9092
REDIS_PORT = 6379
KAFKA_LOCAL_HOST_SERVER = 'kafka:' + str(KAFKA_PORT)
REDIS_HOST = 'redis'
NUM_PIPELINE_INSTANCE = 1
ENCRIPTION_TYPE = 'utf-8'

class Pipeline:
    def __init__(self, instance_id):
        self.instance_id = instance_id
        self.redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                decode_responses=True
                )
        self.converter = Converter()
        self.consumer = KafkaConsumer(
            'rawImageData',
            bootstrap_servers=[KAFKA_LOCAL_HOST_SERVER],
            group_id='image_to_vector_consumer',
            auto_offset_reset='earliest'
        )
        self.producer = KafkaProducer(
            bootstrap_servers=[KAFKA_LOCAL_HOST_SERVER],
            value_serializer=lambda v: json.dumps(v).encode(ENCRIPTION_TYPE)
        )

    def _convert_data(self, data):
        # bug here, still don't know what kind of data type needed by the translation model
        # currently returning a list of float32
        result = self.converter.point_detection(data)
        keypoints = self.converter.extract_keypoints(result)
        keypoints = self.converter.process_new_frame(keypoints)
        return keypoints

    def start_consuming(self):
        print(f'Pipeline {self.instance_id}: Listening for data...')
        for mssg in self.consumer:
            mssg_value = mssg.value
            if mssg.key is None:
                print(f'[{self.instance_id}] Skipping message: \
                      No client ID (key) provided.')
                continue

            if mssg_value is None:
                print('There is no message to receive, skipping...')
                continue

            decoded_value = None
            if isinstance(mssg_value, bytes):
                try:
                    decoded_value = mssg_value.decode(ENCRIPTION_TYPE)
                except UnicodeDecodeError:
                    decoded_value = None

            client_uuid = mssg.key.decode(ENCRIPTION_TYPE)

            if decoded_value == 'stop':
                self._convert_vector_to_words(client_uuid)
                self.redis_client.delete(client_uuid)
                continue

            keypoints = self._convert_data(mssg.value)
            word_prediction = self.translator_model_pred(client_uuid)
            prediction_label, prediction_conf = self.converter.post_process_keypoints(word_prediction)
            print(f'The predicted label is {prediction_label} with {prediction_conf} confidence.')
            self.redis_client.rpush(client_uuid, json.dumps(prediction_label))

    def _translator_model_pred(self, client_uuid):
        # send post request to the ai model server
        pass


if __name__ == '__main__':
    r = redis.Redis(REDIS_HOST, REDIS_PORT)
    r.flushdb()
    print('Redis flush for testing.')

    threads = []

    for i in range(NUM_PIPELINE_INSTANCE):
        pipeline_instance = Pipeline(f'Worker-{i+1}')

        thread = threading.Thread(target=pipeline_instance.start_consuming)
        threads.append(thread)
        thread.start()

    try:
        for thread in threads:
            thread.join()

    except KeyboardInterrupt:
        print('\nShutting down pipelines...')






