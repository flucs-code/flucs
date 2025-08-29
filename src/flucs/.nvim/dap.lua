local dap = require("dap")

local cwd_history = { vim.fn.getcwd() }

local function debug_flucs()
  local choices = vim.deepcopy(cwd_history)
  table.insert(choices, 1, "[New...]")

  vim.ui.select(choices, { prompt = "Select cwd:" }, function(choice)
    local cwd
    if choice == "[New...]" then
      cwd = vim.fn.input("New cwd: ", vim.fn.getcwd(), "dir")
      if cwd ~= "" then
        table.insert(cwd_history, cwd)
      else
        cwd = vim.fn.getcwd()
      end
    else
      cwd = choice
    end

    -- Actually launch dap here
    dap.run({
      type = "python",
      request = "launch",
      name = "Debug flucs (choose cwd)",
      program = vim.fn.exepath("flucs"),
      cwd = cwd,
      justMyCode = false,
    })
  end)
end

-- LSP-friendly keymap using function reference
vim.keymap.set("n", "<leader>dm", debug_flucs, { noremap = true, silent = true, desc = "Debug flucs" })

