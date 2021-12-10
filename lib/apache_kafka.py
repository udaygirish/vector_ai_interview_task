from sys import api_version
from kafka import KafkaConsumer
from kafka import KafkaProducer
from kafka.errors import KafkaError
import time

#Have to implement multi processing later
import multiprocessing as mp
#Kafka Producer can be used across threads without issue
#Kafka consumer can be used in a thread-local manner, multiprocessing is recommended.
import logging 
logger = logging.getLogger(__name__)

class ApacheKafka():

    def __init__(self):
        
        self.description = "This is the base class for Apache Kafka Publisher Subscriber send and receive methods"
        self.boot_strap_server = 'localhost:9092'

    def connect_kafka_producer(self):
        try:
            producer = KafkaProducer(bootstrap_servers=[self.boot_strap_server], api_version=(0,10))
        except Exception as ex:
            logger.error("Exception Caught - Unable to connect to Kafka")
            logger.error("Exception details:{}".format(ex))
        finally:
            self._producer =  producer

            
    def publish_message(self,topic_name, key,value):
        try:
            key_bytes = bytes(key, encoding='utf-8')
            value_bytes = bytes(value, encoding='utf-8')
            self._producer.send(topic_name, key=key_bytes, value = value_bytes)
            self._producer.flush()
            logger.info("Message Published Successfully")
            return 1
        except Exception as ex:
            logger.error('Exception Caught: Message Publishing not successfull')
            logger.error("Exception details:{}".format(ex))
            return 0

    def receive_message(self, topic_name, auto_offset = 'earliest', timeout_ms = 100):
        self.consumer = KafkaConsumer(topic_name, auto_offset_rest = auto_offset,
                                    bootstrap_servers =[self.boot_strap_server], api_version=(0,10), consumer_timeout_ms= timeout_ms)

        consumed_messages = []
        for message in self.consumer:
            temp_message_dict = dict()
            temp_key = message.key.decode('utf-8')
            temp_value = message.value.decode('utf-8')
            temp_offset = message.offset
            temp_message_dict['key'] = temp_key
            temp_message_dict['value'] = temp_value
            temp_message_dict['offset'] = temp_offset
            consumed_messages.append(temp_message_dict)
        self.consumer.close()
        time.sleep(2)
        return consumed_messages
    

