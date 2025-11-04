from kafka import KafkaProducer, KafkaConsumer
from Converter import Converter
import threading
import redis
import socket
import json
import requests

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

    def _convert_data(self, image_bytes):
        keypoints = self.converter.point_detection(image_bytes)
        window = self.converter.process_new_frame(keypoints)
        return window

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
                    decoded_value = mssg_value
            else:
                continue

            client_uuid = mssg.key.decode(ENCRIPTION_TYPE)

            if decoded_value == 'stop':
                self.converter.stop()
                self._convert_to_english_sentence(client_uuid)
                self.redis_client.delete(client_uuid)
                continue

            word_prediction = self._translator_model_pred(decoded_value)
            print('PREDICTION:', word_prediction)
            if word_prediction is None:
                print('WORD PREDICTION IS NONE')
                continue
            prediction_label, prediction_conf = self.converter.post_process_keypoints(word_prediction)
            print(f'The predicted label is {prediction_label} with {prediction_conf} confidence.')
            self.redis_client.rpush(client_uuid, json.dumps(prediction_label))

    def _convert_to_english_sentence(self, uuid):
        #NOT YET DONE IMPLEMENTING
        ai_model_url = 'https://baronocasiones-gestura.hf.space/convert-sentence'
        asl_grammar = self.redis_client.get(uuid)
        payload = {'asl_gloss': asl_grammar}
        response = requests.post(ai_model_url, json=payload).text

    def _translator_model_pred(self, keypoints):
        ai_model_url = 'https://baronocasiones-gestura.hf.space/translate'
        keypoints = self._convert_data(keypoints)
        if keypoints is None:
            return 
        print('KEYPOINTS SHAPE:', keypoints.shape)
        print('KEYPOINTS BEFORE SENDING:', keypoints)
        payload = {'window_data': keypoints.tolist()}
        response = requests.post(ai_model_url, json=payload).text
        print('API RESPONSE:', response)
        pred = json.loads(response).get('pred')
        return pred


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


