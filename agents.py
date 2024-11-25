class DomainAgent:
    def __init__(self, summarizer):
        self.summarizer = summarizer

    def process(self, context: str, prompt_template: str) -> str:
        prompt = prompt_template.format(context=context)
        summary = self.summarizer(prompt, max_length=200, min_length=50, do_sample=False)
        return summary[0]['summary_text']


class HealthcareAgent(DomainAgent):
    def process(self, context: str) -> str:
        prompt_template = """
        Summarize the following medical content accurately and concisely:
        {context}
        Summary:
        """
        return super().process(context, prompt_template)


class FinancialAgent(DomainAgent):
    def process(self, context: str) -> str:
        prompt_template = """
        Provide a concise financial summary of the following content:
        {context}
        Summary:
        """
        return super().process(context, prompt_template)


class GeneralAgent(DomainAgent):
    def process(self, context: str) -> str:
        prompt_template = """
        Summarize the following content clearly and concisely:
        {context}
        Summary:
        """
        return super().process(context, prompt_template)
