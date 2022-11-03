#!/usr/bin/env bash
PORT=8080
echo "Port: $PORT"
# POST method predict
curl -d '{
    "content":"Benedict Arnold.  He was possibly the Continental Army’s best battlefield general, and he led the Americans to victory at the pivotal Battle of Saratoga, where he was wounded. Had he succumbed to his wounds, the northeast US likely would be covered with statues of the man.  As it was, he stewed over the lack of recognition of his greatness and eventually sold out his country to the enemy. Now, his name is synonymous with treason."
}'\
-H "Content-Type: text/html" \
-X POST http://127.0.0.1:$PORT