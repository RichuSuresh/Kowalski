import asyncio
import json
import redis

class RedisService:
    def __init__(self, host, port, chatHistoryLimit, decode_responses=True):
        self.client = redis.Redis(host=host, port=port, decode_responses=decode_responses)
        self.chatHistoryLimit = chatHistoryLimit
    
    def getChatHistory(self, guildId, channelId):
        return [json.loads(message) for message in self.client.lrange(str(guildId) +":"+ str(channelId), 0, -1)]

    def channelExists(self, guildId, channelId):
        return self.client.exists(str(guildId) +":"+ str(channelId))
    
    def addToChatHistory(self, guildId, channelId, ollamaMessage, location):
        if location == "head":
            self.client.lpush(str(guildId) +":"+ str(channelId), json.dumps(ollamaMessage))
        elif location == "tail":
            self.client.rpush(str(guildId) +":"+ str(channelId), json.dumps(ollamaMessage))
        self.client.ltrim(str(guildId) +":"+ str(channelId), -self.chatHistoryLimit, -1)
    
    def addReaction(self, guildId, channelId, messageId, reaction):
        redisId = str(guildId) +":"+ str(channelId)
        for i, message in enumerate(self.client.lrange(redisId, 0, -1)):
            message = json.loads(message)
            messageContent = json.loads(message["content"])
            currentId = messageContent["messageID"]
            if messageId == currentId:
                messageContent["reaction"] = [reaction]
                updatedMessage = {"role": message["role"], "content": json.dumps(messageContent), "images": message["images"]}
                self.client.lset(redisId, i, json.dumps(updatedMessage))
                break
    
    def close(self):
        print("Flushing redis...")
        self.client.flushall()
        print("Redis flushed.")
        self.client.close()