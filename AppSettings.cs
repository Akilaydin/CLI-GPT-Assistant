namespace OriGames.AssistantCLI;

public class AppSettings
{
	public string ContextFilePath { get; set; }
	public string ModelPath { get; set; }
	public string SystemPrompt { get; set; }
	
	public override string ToString()
	{
		return $"{nameof(ContextFilePath)}: {ContextFilePath}, {nameof(ModelPath)}: {ModelPath}, {nameof(SystemPrompt)}: {SystemPrompt}";
	}
}
