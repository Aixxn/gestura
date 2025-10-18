from kafka import KafkaProducer, KafkaConsumer
import json
from Converter import Converter
import threading
import redis
import socket

KAFKA_PORT = 9092
REDIS_PORT = 6739
KAFKA_LOCAL_HOST_SERVER = socket.gethostname() + string(KAFKA_PORT)
REDIS_HOST = socket.gethostname()
NUM_PIPELINE_INSTANCE = 1


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
            group_id='image_to_vector',
            auto_offset_reset='earliest'
        )
        self.producer = KafkaProducer(
            bootstrap_servers=[KAFKA_LOCAL_HOST_SERVER],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

    def _convert_data(self, data):
        # but here, still don't know what kind of data type needed by the translation model
        result = self.converter.point_detection(data)
        keypoints = self.converter.extract_keypoints(result)
        return keypoints

    def start_consuming(self):
        print(f'Pipeline {self.instance_id}: Listening for data...')
        for mssg in self.consumer:
            if mssg.key is None:
                print(f'[{self.instance_id}] Skipping message: \
                      No client ID (key) provided.')
                continue

            client_uuid = mssg.key.decode('utf-8')

            if mssg.value.decode('utf-8') == 'stop':
                self._produce_data(client_uuid)
                self.redis_client.delete(client_uuid)
                continue

            keypoints = self._convert_data(mssg.value)
            self.redis_client.rpush(client_uuid, json.dumps(keypoints))

    def _produce_data(self, client_uuid):
        vector_data_json_string = self.redis_client.lrange(client_uuid, 0, -1)
        vector_data = [json.loads(item) for item in vector_data_json_string]
        payload = {'keypoints': vector_data}
        self.producer.send('keypoints',
                           value=payload,
                           key=client_uuid.encode('utf-8'),
                           )
        self.producer.flush()


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






