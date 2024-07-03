using Microsoft.Extensions.Configuration;
using Microsoft.ML.OnnxRuntimeGenAI;

namespace OriGames.AssistantCLI;

public class AssistantCLI
{
    private const string s_settingsFilePath = "settings.json";
    
    private static AppSettings s_appSettings = new();
    
    public static void Main(string[] args)
    {
        if (args.Length == 0)
        {
            Console.WriteLine("Usage: gpt *your question*");

            return;
        }
        
        ReadSettings();

        var command = args[0];
        
        if (command == "clear")
        {            
            Console.WriteLine("Context cleared");

            ClearContext();
            return;
        }
        
        ProcessCommand(string.Join(" ", args));
    }
    
    private static void ReadSettings()
    {
        var builder = new ConfigurationBuilder()
            .SetBasePath(AppDomain.CurrentDomain.BaseDirectory)
            .AddJsonFile(s_settingsFilePath, optional: false, reloadOnChange: true);
      
        var configurationSection = builder.Build().GetSection("Settings");
        
        s_appSettings.ContextFilePath = configurationSection[nameof(AppSettings.ContextFilePath)] ?? throw new InvalidOperationException();
        s_appSettings.ModelPath = configurationSection[nameof(AppSettings.ModelPath)] ?? throw new InvalidOperationException();
        s_appSettings.SystemPrompt = configurationSection[nameof(AppSettings.SystemPrompt)] ?? throw new InvalidOperationException();
    }

    private static void ProcessCommand(string userCommand)
    {
        try
        {
            Console.WriteLine($"Executing command...");
            
            var executionModel = new Model(s_appSettings.ModelPath);
            var tokenProcessor = new Tokenizer(executionModel);
            var context = UpdateContext(userCommand);

            var tokenisedInput = tokenProcessor.Encode(context);

            ExecuteModel(executionModel, tokenisedInput, tokenProcessor, context);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"An error occurred: {ex.Message}");
        }
    }

    private static string UpdateContext(string userCommand)
    {
        var currentContext = LoadContext();
        var newContext = string.IsNullOrWhiteSpace(currentContext)
            ? FormattedPrompt(s_appSettings.SystemPrompt, userCommand)
            : $"{currentContext}{FormattedPrompt(userCommand)}";
        return newContext;
    }

    private static void ExecuteModel(Model model, Sequences tokenisedInput, Tokenizer tokenProcessor, string context)
    {
        var generatorParams = InitialiseGenerator(model, tokenisedInput);
        var generator = new Generator(model, generatorParams);

        SaveContext(context, GenerateResponse(generator, tokenProcessor));
    }

    private static string FormattedPrompt(string userCommand) => $"<|user|>{userCommand}<|end|><|assistant|>";
    private static string FormattedPrompt(string systemCommand, string userCommand) => $"<|system|>{systemCommand}<|end|>{FormattedPrompt(userCommand)}";

    private static GeneratorParams InitialiseGenerator(Model model, Sequences tokenisedInput)
    {
        var genParams = new GeneratorParams(model);
        genParams.SetSearchOption("max_length", 2048);
        genParams.SetSearchOption("past_present_share_buffer", false);
        genParams.SetInputSequences(tokenisedInput);
        return genParams;
    }

    private static string GenerateResponse(Generator generator, Tokenizer tokenProcessor)
    {
        var result = new System.Text.StringBuilder();
        while (!generator.IsDone())
        {
            generator.ComputeLogits();
            generator.GenerateNextToken();
            var outputTokens = generator.GetSequence(0);
            var newNameToken = outputTokens[^1];
            var output = tokenProcessor.Decode(new []{newNameToken});
            Console.Write(output);
            result.Append(output);
        }

        result.Insert(0, "Assistant: ");
        
        return result.ToString();
    }

    private static void SaveContext(string context, string response)
    {
        context += response;
        File.WriteAllText(s_appSettings.ContextFilePath, context);
    }

    private static void ClearContext() => File.WriteAllText(s_appSettings.ContextFilePath, string.Empty);
        
    private static string LoadContext() => File.Exists(s_appSettings.ContextFilePath) ? File.ReadAllText(s_appSettings.ContextFilePath) : string.Empty;
}