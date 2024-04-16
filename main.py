from dotenv import load_dotenv
from crewai import Crew
from tasks import MeetingPrepTask
from agents import MeetingPrepAgents

def main():
    print("Welcome to CrewAI Meeting Brief!")
    print("---------------------------------")
    meeting_participants = input("What are the email for the participants(other than you) in the meeting?\n")
    meeting_context = input("What is the context of the meeting?\n")
    meeting_objective = input("What is the objective for this meeting?\n")
    
    tasks=MeetingPrepTask()
    agents=MeetingPrepAgents()
    
    #create agents
    research_agent = agents.research_agent()
    industry_analysis_agent = agents.industry_analysis_agent()
    meeting_strategy_agent = agents.meeting_strategy_agent()
    summary_and_briefing_agent = agents.summary_and_briefing_agent()

    #create tasks
    research_task = tasks.research_task(research_agent, meeting_participants, meeting_context)
    industry_analysis_task = tasks.industry_analysis_task(industry_analysis_agent, meeting_participants, meeting_context)
    meeting_strategy_task = tasks.meeting_strategy_task(meeting_strategy_agent, meeting_context, meeting_objective)
    summary_and_briefing_task = tasks.summary_and_briefing_task(summary_and_briefing_agent, meeting_context, meeting_objective)

    meeting_strategy_task.context=[research_task, industry_analysis_task]
    summary_and_briefing_task.context=[research_task, industry_analysis_task, meeting_strategy_task]

    #create crew
    crew=Crew(
        agents=[research_agent, industry_analysis_agent, meeting_strategy_agent, summary_and_briefing_agent],
        tasks=[research_task, industry_analysis_task, meeting_strategy_task, summary_and_briefing_task],
        # verbose=True,
        # process=Process.sequential,
    )
    result=crew.kickoff()
    print(result)

if __name__ == "__main__":
    load_dotenv()
    main()