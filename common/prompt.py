from llama_index import Prompt, PromptTemplate
from llama_index.prompts import PromptType

CH_TEXT_QA_PROMPT_TMPL = (
    "上下文信息如下.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "请注意：在回答时仅依据给出的上下文即可，不要依据你已有的知识来回答。\n"
    "请注意：在回答时仅依据给出的上下文即可，不要依据你已有的知识来回答。\n"
    "请注意：在回答时仅依据给出的上下文即可，不要依据你已有的知识来回答。\n"
    "如果你对答案不自信的话，请回答我不知道。\n"
    "请用中文回答以下问题: {query_str}\n"
)
CH_TEXT_QA_PROMPT = Prompt(
    CH_TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER
)

# # single choice
CH_QUERY_PROMPT_TMPL = (
    "下面给出了一些选择，以编号列表的形式提供。"
    "(1 到 {num_chunks}), "
    "其中列表中的每个项目对应一个摘要。\n"
    "---------------------\n"
    "{context_list}"
    "\n---------------------\n"
    "仅使用上述选项而不依赖先前知识，返回与问题 '{query_str}' 最相关的选择。"
    "以以下格式提供答案：'ANSWER: <number>' 并解释为什么选择该摘要与问题相关。\n"
)
CH_QUERY_PROMPT = PromptTemplate(
    CH_QUERY_PROMPT_TMPL, prompt_type=PromptType.TREE_SELECT
)

CH_CHOICE_SELECT_PROMPT_TMPL = (
    "以下显示了一系列文档。每个文档旁边都有一个数字，"
    "以及文档的摘要。也提供了一个问题。\n"
    "请回答你认为应该参考哪些文档来回答这个问题，按照相关性的顺序，\n"
    "并给出相关性的评分。相关性评分是一个1-10的数字，"
    "根据你认为文档对问题的相关性进行评定。\n"
    "不要包含任何与问题无关的文档。\n"
    "示例格式：\n"
    "Document 1:\n<文档1的摘要>\n\n"
    "Document 2:\n<文档2的摘要>\n\n"
    "...\n\n"
    "Document 10:\n<文档10的摘要>\n\n"
    "问题：<问题>\n"
    "答案：\n"
    "Doc: 9, Relevance: 7\n"
    "Doc: 3, Relevance: 4\n"
    "Doc: 7, Relevance: 3\n\n"
    "现在我们试试：\n\n"
    "{context_str}\n"
    "问题：{query_str}\n"
    "答案：\n"
)
CH_CHOICE_SELECT_PROMPT = PromptTemplate(
    CH_CHOICE_SELECT_PROMPT_TMPL, prompt_type=PromptType.CHOICE_SELECT
)

CH_TREE_SUMMARIZE_TMPL = (
    "以下是来自多个来源的上下文信息。\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "根据来自多个来源的信息，而非先前的知识，"
    "回答下列查询。\n"
    "查询：{query_str}\n"
    "答案："
)
CH_TREE_SUMMARIZE_PROMPT = PromptTemplate(
    CH_TREE_SUMMARIZE_TMPL, prompt_type=PromptType.SUMMARY
)

CH_SINGLE_SELECT_PROMPT_TMPL = (
    "以下给出了一些选项。它们被编号为从1到{num_choices}的列表，"
    "列表中的每一项对应一个摘要。\n"
    "---------------------\n"
    "{context_list}"
    "\n---------------------\n"
    "只使用上述选项，而不使用先前的知识，返回"
    "与问题：'{query_str}'最相关的选项\n"
)

CH_QA_GENERATE_PROMPT_TMPL = """\
以下是上下文信息。

---------------------
{context_str}
---------------------

考虑到上下文信息和没有先验知识。
仅根据以下查询生成问题。

你是一位教师/教授。你的任务是为即将到来的\
{num_questions_per_chunk}个问题的测验/考试设定问题。问题应该在文档中多样化。将问题限制在\
提供的上下文信息中。
"""

CH_SUMMARY_PROMPT_TMPL = (
    "撰写以下内容的摘要。尽量只使用提供的"
    "信息。"
    "尽量包含尽可能多的关键细节。\n"
    "\n"
    "\n"
    "{context_str}\n"
    "\n"
    "\n"
    'SUMMARY："""\n'
)

CH_SUMMARY_PROMPT = PromptTemplate(
    CH_SUMMARY_PROMPT_TMPL, prompt_type=PromptType.SUMMARY
)

CH_INSERT_PROMPT_TMPL = (
    "下述为上下文信息，以编号列表的形式提供 "
    "(从1到{num_chunks})，"
    "列表中的每一项对应一个摘要。\n"
    "---------------------\n"
    "{context_list}"
    "---------------------\n"
    "考虑到上述上下文信息，这里有一条新的"
    "信息：{new_chunk_text}\n"
    "回答应该更新哪个摘要的编号。"
    "答案应为与问题最相关"
    "摘要的编号。\n"
)
CH_INSERT_PROMPT = PromptTemplate(
    CH_INSERT_PROMPT_TMPL, prompt_type=PromptType.TREE_INSERT
)
