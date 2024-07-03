using Microsoft.ML.OnnxRuntimeGenAI;

namespace AssistantCLI;

public class AssistantCLI
{
    private const string s_contextFilePath = "context.txt";
    private const string s_modelPath =
        @"H:\Phi-3-mini-4k-instruct-onnx\cpu_and_mobile\cpu-int4-rtn-block-32";

    private const string s_systemPrompt = "You are an intelligent and user-friendly CLI AI assistant, specializing in providing precise and efficient search results for users' inquiries. " +
        "Don't use markdown or any formatting" +
        "Your responses should be direct, concise, and strictly limited to the information that the user specifically requested. " +
        "Efficiency and accuracy are your primary objectives - eliminate unnecessary details or suggestions, focusing solely on delivering the information inquired." +
        "If you do not know the answer to a question, just say 'I don't know'." + 
        "If user asks for some command provide one example of using";

    public static void Main(string[] args)
    {
        if (args.Length == 0)
        {
            Console.WriteLine("Usage: ask <your question>");
            return;
        }

        var command = args[0].ToLower();
        
        if (command == "clear")
        {            
            Console.WriteLine("Context cleared");

            ClearContext();
            return;
        }

        ProcessCommand(string.Join(" ", args));
    }

    private static void ProcessCommand(string userCommand)
    {
        try
        {
            Console.WriteLine($"Executing command...");
            
            var executionModel = new Model(s_modelPath);
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
            ? FormattedPrompt(s_systemPrompt, userCommand)
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
        File.WriteAllText(s_contextFilePath, context);
    }

    private static void ClearContext() => File.WriteAllText(s_contextFilePath, string.Empty);
        
    private static string LoadContext() => File.Exists(s_contextFilePath) ? File.ReadAllText(s_contextFilePath) : string.Empty;
}