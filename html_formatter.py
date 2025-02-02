import webbrowser

def generate_html(prompts, responses, output_file="output.html"):
    """Generates a Bootstrap-styled HTML page with prompts and responses."""
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>DeepSeek Model Output</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body class="container mt-4">
        <h1 class="mb-4 text-primary">DeepSeek Model Output</h1>
        <div class="list-group">
    """

    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        html_content += f"""
            <div class="list-group-item">
                <h5 class="mb-1">Prompt {i+1}: <span class="text-secondary">{prompt}</span></h5>
                <p class="mb-1"><strong>Response:</strong> {response}</p>
            </div>
        """

    html_content += """
        </div>
    </body>
    </html>
    """

    # Save to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n✅ Responses saved to {output_file}")

    # Open in browser
    webbrowser.open(output_file)
