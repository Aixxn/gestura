from kafka import KafkaProducer, KafkaConsumer
from Converter import Converter
import threading
import redis
import json
import requests

KAFKA_PORT = 9092
REDIS_PORT = 6379
KAFKA_LOCAL_HOST_SERVER = f'kafka:{KAFKA_PORT}'
REDIS_HOST = 'redis'
NUM_PIPELINE_INSTANCE = 1
ENCRYPTION_TYPE = 'utf-8'


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
            value_serializer=lambda v: json.dumps(v).encode(ENCRYPTION_TYPE)
        )

    def _convert_data(self, image_bytes):
        keypoints = self.converter.point_detection(image_bytes)
        window = self.converter.process_new_frame(keypoints)
        return window

    def start_consuming(self):
        print(f'Pipeline {self.instance_id}: Listening for data...')
        for mssg in self.consumer:
            client_uuid = None
            try:
                if mssg.key is None or mssg.value is None:
                    continue

                client_uuid = mssg.key.decode(ENCRYPTION_TYPE)
                value = mssg.value

                # decode if bytes
                if isinstance(value, bytes):
                    try:
                        decoded_value = value.decode(ENCRYPTION_TYPE)
                    except UnicodeDecodeError:
                        decoded_value = value
                else:
                    decoded_value = value

                # STOP SIGNAL
                if isinstance(decoded_value, str) and decoded_value.strip().lower() == 'stop':
                    print(f"[{client_uuid}] Stop signal received. Finalizing sentence...")
                    final_window = self.converter.stop()
                    english_sentence = self._convert_to_english_sentence(client_uuid)
                    print(f"SENTENCE: {english_sentence}", flush=True)
                    self._publish_translated_sentence(client_uuid, english_sentence)
                    self.redis_client.delete(client_uuid)
                    continue

                # otherwise process frame
                word_prediction = self._translator_model_pred(decoded_value)
                if word_prediction is None:
                    continue
                self.redis_client.rpush(client_uuid, word_prediction)

            except Exception as e:
                print(f"[{self.instance_id}] Error processing message for {client_uuid}: {e}")

    def _publish_translated_sentence(self, uuid: str, english_sentence: str):
        """
        Publish the translated English sentence to the Kafka topic 'translatedSign'.
        """
        try:
            message = {
                "uuid": uuid,
                "sentence": english_sentence
            }
            print('PUBLISHING:', message, flush=True)

            # Push to Kafka
            self.producer.send(
                'translatedSign',
                key=uuid.encode(ENCRYPTION_TYPE),
                value=message
            )
            self.producer.flush()

            print(f"[{uuid}] Published to Kafka topic '{translatedSign}': {message}", flush=True)

        except Exception as e:
            print(f"[{uuid}] Error publishing translated sentence to Kafka: {e}")


    def _convert_to_english_sentence(self, uuid):
        ai_model_url = 'https://baronocasiones-gestura.hf.space/convert-sentence'
        word_list = self.redis_client.lrange(uuid, 0, -1)

        if not word_list:
            print(f"[{uuid}] No words found in Redis.")
            return None

        asl_grammar = " ".join(word_list)
        payload = {'asl_gloss': asl_grammar}

        try:
            print("TRANSLATING TO SENTENCE")
            response = requests.post(ai_model_url, json=payload)
            if response.status_code == 200:
                english_sentence = response.json()
                return english_sentence.get('translated', None)
            else:
                print(f"[{uuid}] Failed to convert to sentence: {response.status_code}")
        except Exception as e:
            print(f"[{uuid}] Error contacting sentence model: {e}")

    def _translator_model_pred(self, keypoints):
        ai_model_url = 'https://baronocasiones-gestura.hf.space/translate'
        window = self._convert_data(keypoints)
        if window is None:
            return None

        try:
            payload = {'window_data': window.tolist()}
            print('TRANSLATING TO WORDS')
            response = requests.post(ai_model_url, json=payload)
            response_json = json.loads(response.text)

            pred = response_json.get('pred', None)
            print('PREDICTION:', pred, flush=True)
            return pred
        except Exception as e:
            print(f"[_translator_model_pred] Error: {e}")
            return None


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

