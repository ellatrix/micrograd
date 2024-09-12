module Jekyll
    module ReplaceScriptWithPreFilter
        def replace_script_with_pre(input)
        input.gsub( /<script(.*?)>(.*?)<\/script>/m ) do
            attributes = $1
            content = $2
            num_newlines = content.scan("\n").count
            "<textarea disabled rows='#{num_newlines}'#{attributes}>#{content}</textarea>"
          end
        end
    end
end

Liquid::Template.register_filter(Jekyll::ReplaceScriptWithPreFilter)
