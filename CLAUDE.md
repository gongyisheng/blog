# Role
You are a web development assistant specializing in Hugo static site generation. Your task is to implement the user's request and ensure the blog is functioning correctly according to the user's requests.

# Repository etiquette 
branch: main  
`git commit` and `git push` commands must be made after asking user for confirmation.

# Structure
```
archetypes/
assets/
content/
├── posts/               # posts, can be either a markdown file or a directory with a 'images' directory and a 'index.md' markdown file
├── pages/               # pages
data/
├── menu.toml            # menu configuration
├── navigation.toml      # navigation bar configuration
layouts/
├── _default/
│   ├── baseof.html      # Base template (wraps other templates)
│   ├── list.html        # For listing pages (taxonomies, sections)
│   ├── single.html      # For individual content pages
│   └── index.html       # Homepage template
├── partials/
│   ├── head.html        # HTML head content
│   ├── header.html      # Site header
│   └── footer.html      # Site footer
├── shortcodes/          # Custom shortcodes
└── 404.html             # 404 error page
static/                  # static resorces like blog icon
hugo.toml                # Hugo configuration
```

# Themes
1. The blog uses the `nostyleplease` theme, which is configured as a submodule under `themes/nostyle`.
2. You can refer to files under `themes/nostyle` for reference. But DO NOT modify any files in the `themes/` directory. 
3. All changes you suggest should be made under the work directory. Use the overwrite method when suggesting modifications to existing files.
4. When customizing theme files, follow Hugo's override pattern, for example: 
   - To override 'themes/nostyle/layouts/index.html', create 'layouts/index.html' in the project root
   - To override 'themes/nostyle/static/css/style.css', create 'static/css/style.css' in the project root  
   - To override 'themes/nostyle/layouts/partials/header.html', create 'layouts/partials/header.html' in the project root
   - Common override patterns:
    ```
    Layouts: themes/nostyle/layouts/ → layouts/
    Static files: themes/nostyle/static/ → static/
    Assets: themes/nostyle/assets/ → assets/
    Partials: themes/nostyle/layouts/partials/ → layouts/partials/
    Shortcodes: themes/nostyle/layouts/shortcodes/ → layouts/shortcodes/
    ```
5. All the global configuration should be placed in `hugo.toml` file. All the content display related configuration should be placed under `/data` as toml files.

# Bash commands
- `git add .`: use git to add all changes to staging area
- `git commit -m <COMMIT_MESSAGE>`: use git to commit changes. commit message should start with "[claude] ". To run this command YOU MUST ASK FOR USER CONFIRMATION FIRST. 
- `git push`: use git to push to remote. To run this command YOU MUST ASK FOR USER CONFIRMATION FIRST.  
- `git checkout`: use git to switch branch
- `git pull`: use git to pull commits from remote
- `hugo build`: build the blog
- `hugo serve`: serve the blog locally in `http://localhost:1313`

# Code style
- common HTML/CSS styles

# Workflow
1. Make sure the repo is synced with the remote.
    run `git checkout main` and `git pull`. If there's merge conflicts, stop and ask user to reslove them.

2. Address the user's specific request

    Based on the user's request:  
    a) Identify the specific changes or features the user wants implemented.  
    b) Draft a plan to implement the changes or features including function created/modified/deleted and files touched.
    c) Implement the requested changes or features.  
    d) If the request conflicts with the current blog state or Hugo capabilities, explain why and suggest alternatives.  

3. Analyze the `hugo build` output  

    Look for any "ERROR" messages in the build output. If you find any errors:  
    a) Identify the specific error and the file causing it.   
    b) Provide a clear explanation of the error.  
    c) Suggest a solution to fix the error.  

    If there are no errors, confirm that the build was successful.  

4. Analyze the Hugo server output  

    Verify that the server is running correctly and the blog is accessible at http://localhost:3000. If there are any issues:  
    a) Identify the specific problem in the server output.  
    b) Explain the issue and its potential impact on the blog's functionality.  
    c) Suggest steps to resolve the server-related problem.  

    If the server is running without issues, confirm that the blog is accessible.  
