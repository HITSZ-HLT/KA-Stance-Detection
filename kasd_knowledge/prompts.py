discoures_knowledge_prompt = """The following is a [Sentence] from Twitter about the topic "%s". Please expand the abbreviations, slang and hash tags into complete sentences to restate the [Sentence]. Please give your answer in json format and do not output anything unrelated to the task.
Sentence: "%s"
{"Restated Sentence": "<Your restated sentence>"}"""

episodic_knowledge_prompt = """
If this [Wikipedia Document] is not related to the given [Sentence] and the given [Target], directly output "None". Otherwise, summarize the sentences from the [Wikipedia Document] which related to the given [Sentence] and the given [Target]. Please give your answer in json format and do not output anything unrelated to the task.
Sentence: "%s"
Target: "%s"
Wikipedia Document: "%s"
{"Output": "<Knowledge from [Wikipedia Document] related to the given [Sentence] and the given [Target] / None>"}"""